import anndata
import pandas as pd
from typing import Literal

def get_glm_results(glm_results_list: list, key: Literal["p_value", "statistic"] = "p_value"):
    """
    Get p-values or effect sizes from likelihood ratio test
    The csv file has to contain the columns "event_name" and "p_value"

    Args:
        glm_results_list: list
            list of paths to csv files
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

    glm_results = dd.read_csv(glm_results_list, include_path_column = True)\
        .pivot_table(index = "intron_group", columns = "path", values = key).compute()
    glm_results.rename(columns = {path: Path(path).stem for path in glm_results.columns}, inplace = True)
    glm_results = glm_results.dropna()
    #TODO Why does fdrcorrection turn quantas p values all into 1?
    glm_results = pd.DataFrame(
        fdrcorrection(glm_results.values.flatten())[1].reshape(glm_results.shape), 
        index = glm_results.index, 
        columns = glm_results.columns)
    return glm_results

def main():
    adata = anndata.read_h5ad(snakemake.input.adata_path)
    adata.uns["simple"] = get_glm_results(snakemake.input.simple_result_list, "p_value")
    adata.uns["multiple"] = get_glm_results(snakemake.input.multiple_result_list, "p_value")
    with open(snakemake.output[0], "wb") as f:
        import pickle
        pickle.dump(adata, f)

if __name__ == "__main__":
    main()