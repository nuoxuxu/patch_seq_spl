import scanpy as sc
import pandas as pd
from pathlib import Path
import numpy as np
import src.differential_splicing as ds
import os
import platform
import json
import random

configfile: "config/config.yaml"
localrules: preprocess, download_metadata, download_manifest, download_cpm, generate_bam_list

ephys_props = pd.read_csv("data/ephys_data_sc.csv").columns[1:].to_list()
continuous_predictors = ephys_props + ["soma_depth", "cpm"]
categorical_predictors = ['Sst', 'Pvalb', 'Vip', 'Lamp5', 'Sncg', 'Serpinf1', 'subclass']
all_predictors = continuous_predictors + categorical_predictors
runtime_dict = {"simple": "5h", "multiple": "24h"}

metadata = pd.read_csv('data/20200711_patchseq_metadata_mouse.csv')
with open("data/mappings/transcriptomics_sample_id_file_name.json", "r") as f:
    transcriptomics_sample_id_file_name = json.load(f)
metadata["filename"] = metadata.transcriptomics_sample_id.map(transcriptomics_sample_id_file_name)
metadata.dropna(subset=["filename"], inplace=True)
metadata["full_path"] = metadata["filename"].apply(lambda x: "".join(["/external/rprshnas01/netdata_kcni/stlab/Nuo/STAR_for_SGSeq/coord_bams/", x, "Aligned.sortedByCoord.out.bam"]) if x else None)
metadata["T-type Label"] = metadata["T-type Label"].map(lambda x: "_".join(x.split(" ")))

# rule all:
#     input:
#         expand("proc/quantas/beta_binomial/{predictor}.csv", predictor=ephys_props)

# rule all:
#     input:
#         expand("proc/{group_by}/simple/{predictor}.csv", group_by=["three", "five"], predictor=categorical_predictors, allow_missing=True)

# rule all:
#     input:
#         expand("proc/merge_bams/{cell_type}.bam", cell_type=metadata["T-type Label"].unique())

rule all:
    input:
        expand("proc/merge_bams/{cell_type}.bam.bai", cell_type=metadata["T-type Label"].unique())        

rule download_metadata:
    output: "data/20200711_patchseq_metadata_mouse.csv"
    shell: "curl -O {output} https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/0f/86/0f861fcb-36d5-4d3a-80e6-c3c04f34a8c7/20200711_patchseq_metadata_mouse.csv"

rule download_manifest:
    output: "data/2021-09-13_mouse_file_manifest.csv"
    conda: "patch_seq_spl"
    shell:
        """
        curl https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/81/1d/811d176c-9e06-431b-9691-427edbb6bbd7/2021-09-13_mouse_file_manifest.zip -o 2021-09-13_mouse_file_manifest.zip
        unzip -a 2021-09-13_mouse_file_manifest.zip
        xlsx2csv 2021-09-13_mouse_file_manifest.xlsx > {output}
        rm -f 2021-09-13_mouse_file_manifest.zip 2021-09-13_patchseq_file_download_instructions.docx 2021-09-13_mouse_file_manifest.xlsx
        """

rule download_cpm:
    output: "data/20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
    shell:
        """
        curl https://data.nemoarchive.org/other/AIBS/AIBS_patchseq/transcriptome/scell/SMARTseq/processed/analysis/20200611/20200513_Mouse_PatchSeq_Release_cpm.v2.csv.tar -o 20200513_Mouse_PatchSeq_Release_cpm.v2.csv.tar
        tar -xvf 20200513_Mouse_PatchSeq_Release_cpm.v2.csv.tar
        rm -f 20200513_Mouse_PatchSeq_Release_cpm.v2.csv.tar
        mv 20200513_Mouse_PatchSeq_Release_cpm.v2/20200513_Mouse_PatchSeq_Release_cpm.v2.csv {output}
        rm -rf 20200513_Mouse_PatchSeq_Release_cpm.v2
        """

rule tabs_to_adata:
    input: 
        SJ_out_tabs="data/SJ_out_tabs",
        metadata="data/20200711_patchseq_metadata_mouse.csv",
        manifest="data/2021-09-13_mouse_file_manifest.csv",
        gtf_path=Path(os.environ["GENOMIC_DATA_DIR"]).joinpath("Ensembl/Mouse/Release_110/Raw/Mus_musculus.GRCm39.110.gtf")        
    output: "proc/scquint/adata_{group_by}.h5ad"
    resources: 
        runtime = "1h"
    script: "scripts/tabs_to_adata.py"

# group_by is either "three" or "five"
rule preprocess:
    input: 
        adata_path="proc/scquint/adata_{group_by}.h5ad",
    output: "proc/scquint/preprocessed_adata_{group_by}.h5ad"
    script: "scripts/preprocess.py"

# predictor is "cpm", "soma_depth" and any of the ephys props
# model is either "simple", "multiple" or "categorical"
rule run_GLM:
    input: "proc/scquint/preprocessed_adata_{group_by}.h5ad"
    output: "proc/scquint/{group_by}/{model}/{predictor}.csv"
    resources:
        runtime = lambda wildcards: runtime_dict[wildcards.model]
    script: "scripts/run_GLM.py"

rule Fig1_heatmap:
    script: "scripts/Fig1_heatmap.py"

rule beta_binomial:
    output:
        "proc/quantas/{statistical_model}/{predictor}.csv"
    resources:
        runtime="1h"
    conda: "test_arrow"
    shell:
        "Rscript scripts/beta_binomial.R {wildcards.predictor} {wildcards.statistical_model}"

################# Merge BAMs #################
rule generate_bam_list:
    output:
        "proc/merge_bams/{cell_type}.txt"
    script:
        "scripts/generate_bam_list.py"

rule merge_bams:
    input:
        "proc/merge_bams/{cell_type}.txt"
    output:
        "proc/merge_bams/{cell_type}.bam"
    resources:
        runtime=120,
        mem_mb=50000,
        threads=12
    shell:
        "samtools merge -o {output} -b {input} -@ 8"

rule index_bams:
    input:
        "proc/merge_bams/{cell_type}.bam"
    output:
        "proc/merge_bams/{cell_type}.bam.bai"
    resources:
        runtime=60,
        mem_mb=150000,
        threads=12
    shell:
        "samtools index {input} -@ 8"