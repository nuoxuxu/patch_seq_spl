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

# group_by = ["three", "five"]
# ephys_props = pd.read_csv("data/ephys_data_sc.csv").columns[1:].to_list()
# continuous_predictors = ephys_props + ["soma_depth", "cpm"]
# categorical_predictors = ['Sst', 'Pvalb', 'Vip', 'Lamp5', 'Sncg', 'Serpinf1', 'subclass']
# with open("data/mappings/transcriptomics_file_name_cell_type.json", "r") as f:
#     transcriptomics_file_name_cell_type = json.load(f)
# transcriptomics_file_name_cell_type = {k: v.replace(" ", "_") for k, v in transcriptomics_file_name_cell_type.items()}         
# cell_types = list(set(transcriptomics_file_name_cell_type.values()))
# all_predictors = continuous_predictors + categorical_predictors
# runtime_dict = {"simple": "3h", "multiple": "24h"}

# metadata = pd.read_csv('data/20200711_patchseq_metadata_mouse.csv')
# with open("data/mappings/transcriptomics_sample_id_file_name.json", "r") as f:
#     transcriptomics_sample_id_file_name = json.load(f)
# metadata["filename"] = metadata.transcriptomics_sample_id.map(transcriptomics_sample_id_file_name)
# metadata.dropna(subset=["filename"], inplace=True)
# metadata["full_path"] = metadata["filename"].apply(lambda x: "".join(["proc/star/", x, ".Aligned.sortedByCoord.out.bam"]) if x else None)
# metadata["T-type Label"] = metadata["T-type Label"].map(lambda x: "_".join(x.split(" ")))

# rule all:
#     input:
#         expand("proc/quantas/beta_binomial/{predictor}.csv", predictor=ephys_props)

# rule all:
#     input:
#         expand("proc/scquint/three/simple/{predictor}.csv", predictor=ephys_props, allow_missing=True)

dataset="sorensen"
sample_list = [path.name for path in Path(f"data/sorensen/transcriptome").iterdir()][0]

rule all:
    input:
        expand("proc/star/{dataset}/{sample}.SJ.out.tab", dataset=dataset, sample=sample_list)

# rule all:
#     input:
#         expand("proc/merge_bams/{cell_type}.bam", cell_type=metadata["T-type Label"].unique())

# rule all:
#     input:
#         expand("proc/merge_bams/{cell_type}.bam.bai", cell_type=metadata["T-type Label"].unique())

rule create_sample_sheet:
    output:
        "proc/sample_sheet_{dataset}.csv"
    script:
        "scripts/sample_sheet.py"

rule star_index:
    input:
        fasta=Path(os.environ["GENOMIC_DATA_DIR"]).joinpath("Ensembl/Mouse/Release_110/Raw/Mus_musculus.GRCm39.dna.primary_assembly.fa"),
        gtf=Path(os.environ["GENOMIC_DATA_DIR"]).joinpath("Ensembl/Mouse/Release_110/Raw/Mus_musculus.GRCm39.110.gtf")
    output:
        directory(Path(os.environ["GENOMIC_DATA_DIR"]).joinpath("Ensembl/Mouse/Release_110/STAR_index")),
    message:
        "Testing STAR index"
    log:
        "logs/star_index.log",
    shell:
        """
        STAR \
            --runThreadN {threads} \
             --runMode genomeGenerate \
             --genomeDir {output} \
             --genomeFastaFiles {input.fasta} \
             --sjdbGTFfile {input.gtf}
        """

def get_fastq(wildcards):
    base_dir = Path("data") / wildcards.dataset / "transcriptome" / wildcards.sample
    r1 = base_dir / f"{wildcards.sample}_R1.fastq.gz"
    r2 = base_dir / f"{wildcards.sample}_R2.fastq.gz"
    return {"r1": r1, "r2": r2}

rule star_align:
    input:
        unpack(get_fastq),
        index=Path(os.environ["GENOMIC_DATA_DIR"]).joinpath("Ensembl/Mouse/Release_110/STAR_index")
    output:
        bam="proc/star/{dataset}/{sample}.Aligned.sortedByCoord.out.bam",
        SJ_out_tab="proc/star/{dataset}/{sample}.SJ.out.tab",
        gene_counts="proc/star/{dataset}/{sample}.ReadsPerGene.out.tab"
    params:
        out_prefix="proc/star/{dataset}/{sample}."
    log:
        "logs/star_align/{dataset}/{sample}.log",
    shell:
        """
        STAR \
            --runThreadN {threads} \
             --genomeDir {input.index} \
             --readFilesIn {input.r1} {input.r2} \
             --readFilesCommand zcat \
             --outFileNamePrefix {params.out_prefix} \
             --outSAMtype BAM SortedByCoordinate \
             --outSAMunmapped Within \
             --quantMode GeneCounts
        """

rule tabs_to_adata:
    input: 
        sample_sheet="proc/sample_sheet.csv"
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

# rule run_GLM:
#     input: "proc/scquint/preprocessed_adata_{group_by}.h5ad"
#     output: "proc/scquint/{group_by}/{model}/{predictor}.csv"
#     resources:
#         runtime = lambda wildcards: runtime_dict[wildcards.model]
#     script: "scripts/run_GLM.py"

# rule add_glm:
#     input:
#         adata_path="proc/scquint/preprocessed_adata_{group_by}.h5ad",
#         simple_result_list=expand("proc/scquint/{{group_by}}/simple/{predictor}.csv", predictor=ephys_props),
#         multiple_result_list=expand("proc/scquint/{{group_by}}/multiple/{predictor}.csv", predictor=ephys_props)
#     output:
#         "results/scquint/{group_by}.pkl"
#     script: "scripts/add_glm_results.py"

# rule Fig1_heatmap:
#     script: "scripts/Fig1_heatmap.py"

# rule beta_binomial:
#     output:
#         "proc/quantas/{statistical_model}/{predictor}.csv"
#     resources:
#         runtime="1h"
#     conda: "test_arrow"
#     shell:
#         "Rscript scripts/beta_binomial.R {wildcards.predictor} {wildcards.statistical_model}"

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