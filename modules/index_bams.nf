process INDEX_BAMS {
    tag "${cell_type}"
    publishDir "${params.output_dir}/merge_bams", mode: 'copy'
    
    input:
        tuple val(cell_type), path(bam_file)
    
    output:
        tuple val(cell_type), path("${bam_file}.bai"), emit: indexed_bam
    
    script:
    """
    samtools index ${bam_file} -@ ${task.cpus}
    """
}
