import pandas as pd
import anndata as ad
import numpy as np
path_to_metadata = "data/20200711_patchseq_metadata_mouse.csv"
metadata = pd.read_csv(path_to_metadata)
ID_to_ttype = metadata.loc[:, ["cell_specimen_id", "T-type Label"]]
ID_to_ttype = ID_to_ttype[~ID_to_ttype["T-type Label"].isna()] \
    .set_index("cell_specimen_id") \
    .to_dict()["T-type Label"]

path_to_manifest = "data/2021-09-13_mouse_file_manifest.csv"
# From manifest, get filenames of cells that have valid t-type labels (passed QC)
flist = pd.read_csv(path_to_manifest) \
    .query("file_type == 'reverse_fastq'") \
    .query("file_name.str.contains('fastq.gz')", engine = "python") \
    .assign(file_name = lambda x : x.file_name.str.removesuffix("_R2.fastq.gz")) \
    .loc[:, ["file_name", "cell_specimen_id"]] \
    .assign(ttype = lambda x : x.cell_specimen_id.map(ID_to_ttype)) \
    .dropna(subset = ["ttype"]) \
    ["file_name"].to_list()

from src.extended_anndata import *
adata = ad.read_h5ad("proc/scquint/preprocessed_adata_three.h5ad")
adata = ExtendedAnnData(adata)
adata = adata.add_predictors()
adata.obsm["predictors"]

import json
import warnings
from sklearn.impute import SimpleImputer

ephys_data_path = "data/ephys_data_sc.csv"
transcriptomic_ID_subclass_path = "data/mappings/transcriptomic_ID_subclass.json"
with open(transcriptomic_ID_subclass_path, "r") as f:
    transcriptomic_ID_subclass = json.load(f)
with open("data/mappings/transcriptomics_file_name_cell_type.json") as f:
    transcriptomics_file_name_cell_type = json.load(f)
transcriptomic_id_to_specimen_id_path = "data/mappings/transcriptomic_id_to_specimen_id.json"
metadata_path = "data/20200711_patchseq_metadata_mouse.csv"    

ephys_data = pd.read_csv(ephys_data_path, index_col = 0)
ephys_data = ephys_data.loc[(np.isnan(ephys_data)).sum(axis = 1) < 6, :]
print("Removing cells with more than 6 missing ephys properties")
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
ephys_data = pd.DataFrame(imp.fit_transform(ephys_data), index=ephys_data.index, columns=ephys_data.columns)
print("Imputing the rest of the missing ephys values with mean")

# keep only cells that are in both adata and ephys_data    
common_IDs = np.intersect1d(adata.obs_names, ephys_data.index)
adata = adata[common_IDs, :]
ephys_data = ephys_data.loc[common_IDs]

# Add subclass and cell type labels to ephys data
ephys_data = ephys_data.assign(subclass = ephys_data.index.map(transcriptomic_ID_subclass)).dropna()
ephys_data = ephys_data.assign(cell_type = ephys_data.index.map(transcriptomics_file_name_cell_type)).dropna()
ephys_data = ephys_data.assign(cell_type = lambda x: x["cell_type"].str.replace(" ", "_"))

# Add cpms to ephys data
cpm_path = "data/20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
transcriptomics_sample_id_file_name_path = "data/mappings/transcriptomics_sample_id_file_name.json"
transcriptomics_sample_id_file_name = json.load(open(transcriptomics_sample_id_file_name_path, "r"))        
cpm = pd.read_csv(cpm_path, index_col=0)

cpm = cpm.loc[cpm.index.isin(adata.var.gene_name.values.categories), :]
cpm = cpm.T
cpm.index = cpm.index.map(transcriptomics_sample_id_file_name)

ephys_data = pd.concat([ephys_data, cpm.loc[ephys_data[ephys_data.index.isin(cpm.index)].index, :]], axis = 1).dropna()
adata = adata[ephys_data.index, :]

# Add soma depth to ephys data
with open(transcriptomic_id_to_specimen_id_path, "r") as f:
    transcriptomic_id_to_specimen_id = json.load(f)
metadata = pd.read_csv(metadata_path)

specimen_id_to_transcriptomic_id = {v: k for k, v in transcriptomic_id_to_specimen_id.items()} 
metadata = metadata.loc[~metadata["cell_soma_normalized_depth"].isna()]
metadata["cell_specimen_id"] = metadata["cell_specimen_id"].map(specimen_id_to_transcriptomic_id)
cell_soma_normalized_depth = metadata.dropna(subset=["cell_specimen_id"]).set_index("cell_specimen_id")["cell_soma_normalized_depth"]    

ephys_data = ephys_data.assign(soma_depth = ephys_data.index.map(cell_soma_normalized_depth.to_dict()))

for subclass in ephys_data["subclass"].unique():
    ephys_data[subclass] = ephys_data["subclass"] == subclass
for cell_type in ephys_data["cell_type"].unique():
    ephys_data[cell_type] = ephys_data["cell_type"] == cell_type

adata.obsm["predictors"] = ephys_data
from pathlib import Path
adata.uns["simple"] = get_glm_results(glm_results_list=[str(a) for a in Path("proc/scquint/three/simple").iterdir()], key="p_value")

def get_glm_results(glm_results_list: list, key):
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

import polars as pl

(adata.uns["simple"] < 0.05).sum(axis=0)