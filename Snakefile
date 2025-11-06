import pandas as pd

df = pd.read_csv("data/2021-09-13_mouse_file_manifest.csv")
sample_list = (
    df.loc[df["file_type"] == "forward_fastq", "file_name"]
    .str.replace("_R1.fastq.gz", "", regex=False)
    .tolist()
)

# remove problematic sample as in original
sample_list.remove("SM-GBMG3_E1-50_AGTTAGCTGG-CATTCTCATC")

ephys_df = pd.read_csv("data/ephys_data_sc.csv")
ephys_list = ephys_df.columns.tolist()[1:]

rule all: 
    input:
        expand("results/three_ttype_simple/{ephys}/{chunk}.pkl", ephys=ephys_list, chunk=range(1, 60))

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

rule star_align:
    input:
        r1="data/transcriptome/{sample}/{sample}_R1.fastq.gz",
        r2="data/transcriptome/{sample}/{sample}_R2.fastq.gz",
        index=Path(os.environ["GENOMIC_DATA_DIR"]).joinpath("Ensembl/Mouse/Release_110/STAR_index")
    output:
        bam="proc/star/{sample}.Aligned.sortedByCoord.out.bam",
        SJ_out_tab="proc/star/{sample}.SJ.out.tab",
        gene_counts="proc/star/{sample}.ReadsPerGene.out.tab"
    params:
        out_prefix="proc/star/{sample}."
    log:
        "logs/star_align/{sample}.log",
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

rule generate_sample_sheet:
    input:
        SJ_out_tab=expand("proc/star/{sample}.SJ.out.tab", sample=sample_list),
        prefix="proc/star",
        metadata="data/20200711_patchseq_metadata_mouse.csv",
        file_manifest="data/2021-09-13_mouse_file_manifest.csv"
    output:
        "proc/sample_sheet.csv"
    log:
        "logs/sample_sheet.log"
    script: "scripts/sample_sheet.py"

rule preprocess:
    input:
        sample_sheet="proc/sample_sheet.csv",
        file_manifest="data/2021-09-13_mouse_file_manifest.csv",
        metadata="data/20200711_patchseq_metadata_mouse.csv",
        ephys_data="data/ephys_data_sc.csv",
        gtf_file=Path(os.environ["GENOMIC_DATA_DIR"]).joinpath("Ensembl/Mouse/Release_110/Raw/Mus_musculus.GRCm39.110.gtf")
    output:
        "proc/ratio_matrix_{sharing}_{mode}.parquet",
        "proc/ephys_data_{sharing}_{mode}.parquet"
    params:
        sharing=lambda wildcards: wildcards.sharing,
        mode=lambda wildcards: wildcards.mode
    log:
        "logs/preprocess_{sharing}_{mode}.log"
    shell:
        """
        scripts/preprocess.py \
            --sample_sheet {input.sample_sheet} \
            --file_manifest {input.file_manifest} \
            --metadata {input.metadata} \
            --ephys_data {input.ephys_data} \
            --gtf_file {input.gtf_file} \
            --sharing {params.sharing} \
            --mode {params.mode}
        """

rule generate_chunks:
    input:
        ratio_matrix="proc/ratio_matrix_{sharing}_{mode}.parquet"
    output:
        expand("proc/ratio_matrix/{{sharing}}_{{mode}}/{chunk}.txt", chunk=range(1, 61))
    params:
        sharing=lambda wildcards: wildcards.sharing,
        mode=lambda wildcards: wildcards.mode
    log:
        "logs/generate_chunks_{sharing}_{mode}.log"
    shell:
        """
        scripts/generate_chunks.py \
            --sharing {params.sharing} \
            --mode {params.mode} \
            --num_chunks 60
        """

rule run_GLM:
    input:
        "proc/ratio_matrix/{sharing}_{mode}/{chunk}.txt",
        "proc/ratio_matrix_{sharing}_{mode}.parquet",
        "proc/ephys_data_{sharing}_{mode}.parquet"
    output:
        "results/{sharing}_{mode}_{model}/{predictor}/{chunk}.pkl"
    params:
        sharing=lambda wildcards: wildcards.sharing,
        mode=lambda wildcards: wildcards.mode,
        predictor=lambda wildcards: wildcards.predictor,
        model=lambda wildcards: wildcards.model
    log:
        "logs/{sharing}_{mode}_{model}/{predictor}/{chunk}.log"
    shell:
        """
        scripts/run_GLM.py \
            --sharing {params.sharing} \
            --mode {params.mode} \
            --model {params.model} \
            --predictor {params.predictor} \
            --chunk {wildcards.chunk}
        """