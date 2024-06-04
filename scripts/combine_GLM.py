from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import scanpy as sc

full_df = []
for path in Path("proc/three/simple").iterdir():
    df = pd.read_csv(path, index_col=0)
    df = df.assign(ephys_prop = Path(path).name.removesuffix(".csv"))
    full_df.append(df)
full_df = pd.concat(full_df)

# Some rows contain NaNs because the optimization failed
full_df["p_value"] = np.nan_to_num(full_df["p_value"], nan=1)
full_df = full_df.assign(p_adj = fdrcorrection(full_df["p_value"])[0])

pivot_p_value = full_df[["intron_group", "ephys_prop", "p_value"]].pivot(index="intron_group", columns="ephys_prop", values="p_value")

p_adj_df = pd.DataFrame(
    fdrcorrection(np.nan_to_num(pivot_p_value.values.flatten(), nan=1))[1].reshape(pivot_p_value.shape),
    index=pivot_p_value.index,
    columns=pivot_p_value.columns)

p_adj_df.replace(1, np.nan, inplace=True)
p_adj_df = -np.log(p_adj_df)

# set -log10 of NaN pvalues to 0
p_adj_df.replace([np.inf, -np.inf], 0, inplace=True)

adata_GLM = sc.read_h5ad("results/adata_three_simple.h5ad")
adata_GLM.var_names
adata_GLM.var.intron_group

p_adj_df.loc[adata_GLM.var.intron_group.to_list(), :].reset_index(drop=True).index

adata_GLM.varm["corr"] = p_adj_df.loc[adata_GLM.var.intron_group.to_list(), :].reset_index(drop=True)
pivot_fdr = full_df[["intron_group", "ephys_prop", "p_adj"]].pivot(index="intron_group", columns="ephys_prop", values="p_adj")
adata_GLM.varm["fdr"] = pivot_fdr.loc[adata_GLM.var.intron_group.to_list(), :].values
adata_GLM.varm["fdr"] = adata_GLM.varm["fdr"].astype(float)

adata_GLM.write(snakemake.output[0])