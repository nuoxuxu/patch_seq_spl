process STAR_ALIGN {
    tag "${sample}"
    publishDir "${params.output_dir}/star/${dataset}", mode: 'copy', pattern: "*.bam"
    publishDir "${params.output_dir}/star/${dataset}", mode: 'copy', pattern: "*.SJ.out.tab"
    publishDir "${params.output_dir}/star/${dataset}", mode: 'copy', pattern: "*.ReadsPerGene.out.tab"
    
    input:
        tuple val(sample), val(dataset)
        path(star_index)
    
    output:
        tuple val(sample), val(dataset), path("${sample}.Aligned.sortedByCoord.out.bam"), path("${sample}.SJ.out.tab"), path("${sample}.ReadsPerGene.out.tab"), emit: star_output
    
    script:
    def r1 = file("${projectDir}/data/${dataset}/transcriptome/${sample}/${sample}_R1.fastq.gz")
    def r2 = file("${projectDir}/data/${dataset}/transcriptome/${sample}/${sample}_R2.fastq.gz")
    
    """
    STAR \\
        --runThreadN ${task.cpus} \\
        --genomeDir ${star_index} \\
        --readFilesIn ${r1} ${r2} \\
        --readFilesCommand zcat \\
        --outFileNamePrefix ${sample}. \\
        --outSAMtype BAM SortedByCoordinate \\
        --outSAMunmapped Within \\
        --quantMode GeneCounts
    """
}
