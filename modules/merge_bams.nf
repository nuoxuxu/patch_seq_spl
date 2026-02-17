process MERGE_BAMS {
    tag "${cell_type}"
    publishDir "${params.output_dir}/merge_bams", mode: 'copy'
    
    input:
        tuple val(cell_type), path(bam_list)
    
    output:
        tuple val(cell_type), path("${cell_type}.bam"), emit: merged_bam
    
    script:
    """
    samtools merge -o ${cell_type}.bam -b ${bam_list} -@ ${task.cpus}
    """
}
