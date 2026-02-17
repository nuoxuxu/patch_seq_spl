process TABS_TO_ADATA {
    publishDir "${params.output_dir}/scquint", mode: 'copy'
    
    input:
        path(sample_sheet)
        val(group_by)
    
    output:
        path("adata_${group_by}.h5ad"), emit: adata
    
    script:
    """
    tabs_to_adata.py \\
        --sample_sheet ${sample_sheet} \\
        --group_by ${group_by} \\
        --output adata_${group_by}.h5ad
    """
}
