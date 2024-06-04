import anndata
import scquint.data as sd
import src.helper_functions as src
from importlib import reload
import pandas as pd
import scquint.differential_splicing as ds
from pathlib import Path
import json
import numpy as np

reload(ds)

adata = anndata.read_h5ad("proc/adata_filtered.h5ad")
morpho_data = pd.read_csv("data/gouwen_features__wide_normalized2.csv")
with open("/scratch/s/shreejoy/nxu/patch_seq_spl_archive/proc/mappings/transcriptomic_id_to_specimen_id.json", "r") as f:
    transcriptomic_id_to_specimen_id = json.load(f)
specimen_id_to_transcriptomic_id = {v: k for k, v in transcriptomic_id_to_specimen_id.items()}    
morpho_data.specimen_id = morpho_data.specimen_id.map(specimen_id_to_transcriptomic_id)
morpho_data = morpho_data.dropna()
morpho_data = morpho_data.loc[morpho_data.specimen_id.isin(adata.obs_names)]
adata = adata[morpho_data.specimen_id]
del adata.obsm["ephys_prop"]
for morpho_prop in morpho_data.columns:
    adata.obsm[morpho_prop] = morpho_data[morpho_prop].values
bool_array = pd.merge(adata.to_df().T.sum(axis=1).rename("sum"), adata.var["intron_group"], left_index=True, right_index=True).groupby("intron_group").sum() >= 200
intron_group_to_keep = bool_array.where(bool_array).dropna().index.to_numpy()
adata = adata[:, adata.var.intron_group.isin(intron_group_to_keep)]
del adata.obsm["specimen_id"]
adata.var = adata.var.reset_index()
adata.var.index = adata.var.index.astype(str)

df_list = []
for morpho_prop in list(adata.obsm)[:3]:
    column = []
    for intron_group in adata.var.intron_group.unique().to_numpy():
        df, _ = ds.run_regression(adata, intron_group, morpho_prop, subclass=False)    
        column.append(df["p_value"].values)
    column = np.hstack(column)
    column = pd.Series(column, name=morpho_prop)
    df_list.append(column)
df_full = pd.concat(df_list, axis=1)