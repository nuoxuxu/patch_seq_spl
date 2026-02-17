process PREPROCESS {
    publishDir "${params.output_dir}/scquint", mode: 'copy'
    
    input:
        path(adata_path)
    
    output:
        path("preprocessed_${adata_path.baseName}.h5ad"), emit: preprocessed_adata
    
    script:
    """
    python3 ${projectDir}/scripts/preprocess.py \\
        --input ${adata_path} \\
        --output preprocessed_${adata_path.baseName}.h5ad
    """
}
