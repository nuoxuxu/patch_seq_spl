import pandas as pd
import numpy as np
from pathlib import Path
from utility.utils import correlate
import dask.dataframe as dd
from statsmodels.stats.multitest import fdrcorrection
import anndata

def intron_predictor_correlation(adata, intron_group, predictor, intron = False):
    if predictor == "cpm":
        predictor = intron_group.split("_")[0]
            
    cells_to_use_PSI = np.flatnonzero(adata[:, adata.var.intron_group == intron_group].X.toarray().sum(axis = 1) != 0)
    cells_to_use_predictor = np.flatnonzero(~np.isnan(adata.obsm[predictor]))
    cells_to_use = np.intersect1d(cells_to_use_PSI, cells_to_use_predictor)

    psi = adata[:, adata.var.intron_group == intron_group].X.toarray()
    psi = psi[cells_to_use]
    psi = psi / psi.sum(axis=1)[:, None]

    x = adata[:, adata.var.intron_group == intron_group].obsm[predictor].toarray()
    x = x[cells_to_use][:, None]
    
    try:
        correlation = correlate(psi, x, rank = True).correlation
    except AssertionError as e:
        print(e)
        print(f"Something happened with {intron_group} {predictor}")
        correlation = np.repeat(np.nan, psi.shape[1])[:,]

    return correlation

glm_results = dd.read_csv([path for path in Path(snakemake.input[0]).iterdir()], include_path_column = True)\
    .pivot_table(index = "intron_group", columns = "path", values = "p_value").compute()
glm_results.rename(columns = {path: Path(path).stem for path in glm_results.columns}, inplace = True)
glm_results = glm_results.dropna()
glm_results = pd.DataFrame(
    fdrcorrection(glm_results.values.flatten())[1].reshape(glm_results.shape), 
    index = glm_results.index, 
    columns = glm_results.columns)
top_100 = glm_results.loc[glm_results.apply(lambda x: (x < 0.05)).sum(axis = 1).sort_values(ascending = False)[:100].index]
preprocessed_adata = anndata.read_h5ad(snakemake.input[1])

# This code produces a dataframe for each intron_group, the largest correlation value between one of the 
# introns and the predictor in the columns
intron_group_corr_array = []
for intron_group in top_100.index:
    row = []
    for predictor in top_100.columns:
        corr_arr = intron_predictor_correlation(preprocessed_adata, intron_group, predictor).squeeze()
        largest_corr_idx = np.argmax(np.abs(corr_arr))
        row.append(corr_arr[largest_corr_idx])
    row = pd.Series(row, name= intron_group, index = top_100.columns)
    intron_group_corr_array.append(row)
intron_group_corr_array = pd.concat(intron_group_corr_array, axis = 1)
intron_group_corr_array.to_csv(snakemake.output[0])

# This code produces a dataframe for each intron_group, the largest correlation value between one of the 
# introns and the predictor in the columns
intron_corr_array = []
for intron_group in top_100.index:
    predictor_column = []
    for predictor in top_100.columns:
        temp = pd.DataFrame(
            intron_predictor_correlation(preprocessed_adata, intron_group, predictor).squeeze(), 
            index = preprocessed_adata[:, preprocessed_adata.var.intron_group == intron_group].var_names,
            columns = ["correlation"]
            )
        temp = temp.assign(predictor = predictor)
        predictor_column.append(temp)
    predictor_column = pd.concat(predictor_column, axis = 0)
    intron_corr_array.append(predictor_column)
intron_corr_array = pd.concat(intron_corr_array, axis = 0)
intron_corr_array.to_csv(snakemake.output[1])