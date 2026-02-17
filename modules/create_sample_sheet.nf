process CREATE_SAMPLE_SHEET_GOUWENS {
    publishDir "${params.output_dir}", mode: 'copy'
    
    input:
    val(dataset)
    path(metadata)
    path(file_manifest)
    
    output:
    tuple val(dataset), path("sample_sheet_${params.dataset}.csv"), emit: sample_sheet
    
    script:
    """
    sample_sheet_gouwens.py \\
        --dataset $dataset \\
        --metadata $metadata \\
        --file_manifest $file_manifest
    """
}

process CREATE_SAMPLE_SHEET_SORENSEN {
    publishDir "${params.output_dir}", mode: 'copy'
    
    input:
    path(dataset)
    path(metadata)
    path(file_manifest)
    
    output:
    tuple val(dataset), path("sample_sheet_${params.dataset}.csv"), emit: sample_sheet
    
    script:
    """
    sample_sheet_sorensen.py \
        --dataset $dataset \\
        --metadata $metadata \\
        --file_manifest $file_manifest
    """
}
