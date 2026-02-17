process GENERATE_BAM_LIST {
    output:
        tuple val(cell_type), path("${cell_type}_bam_list.txt"), emit: bam_list
    
    script:
    """
    python3 ${projectDir}/scripts/generate_bam_list.py \\
        --cell_type ${cell_type} \\
        --output ${cell_type}_bam_list.txt
    """
}
