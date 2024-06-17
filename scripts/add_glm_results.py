import anndata
import pandas as pd
from typing import Literal

def get_glm_results(path: str, key: Literal["p_value", "statistic"] = "p_value"):
    """
    Get p-values or effect sizes from likelihood ratio test
    The csv file has to contain the columns "event_name" and "p_value"

    Args:
        path: str
            path to directory containing csv files
        key: str
            "p_value" or "statistic"

    Returns:
        glm_results: pd.DataFrame
            adjusted p-values from likelihood ratio test
    """
    import dask.dataframe as dd
    import pandas as pd
    from pathlib import Path
    from statsmodels.stats.multitest import fdrcorrection

    glm_results = dd.read_csv([path for path in Path(path).iterdir()], include_path_column = True)\
        .pivot_table(index = "event_name", columns = "path", values = key).compute()
    glm_results.rename(columns = {path: Path(path).stem for path in glm_results.columns}, inplace = True)
    glm_results = glm_results.dropna()
    #TODO Why does fdrcorrection turn quantas p values all into 1?
    glm_results = pd.DataFrame(
        fdrcorrection(glm_results.values.flatten())[1].reshape(glm_results.shape), 
        index = glm_results.index, 
        columns = glm_results.columns)
    
    if "quantas" in path:
        event_name_gene_name = pd.read_csv("data/quantas/Mm.seq.all.AS.chrom.can.id2gene2symbol", sep = "\t", header = None)\
            .set_index(0)\
            .loc[:, 2]\
            .to_dict()        
        glm_results["gene_name"] = glm_results.index.map(event_name_gene_name)
        glm_results["gene_name"] = glm_results.groupby("gene_name").cumcount().add(1).astype(str).radd(glm_results["gene_name"] + '_')
        glm_results = glm_results.set_index("gene_name")

    return glm_results

for group_by in ["three", "five"]:
    adata = anndata.read_h5ad(f"proc/scquint/preprocessed_adata_{group_by}.h5ad")
    adata.uns["simple"] = get_glm_results(f"proc/scquint/{group_by}/simple")
    adata.uns["multiple"] = get_glm_results(f"proc/scquint/{group_by}/multiple")
    adata.write_h5ad(f"proc/scquint/preprocessed_adata_{group_by}.h5ad")