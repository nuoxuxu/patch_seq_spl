import multiprocessing as mp
import src.helper_functions as src
from importlib import reload
import anndata
import pandas as pd
import scquint.differential_splicing as ds
from functools import partial
import numpy as np
from scipy.sparse import csr_matrix
import src.helper_functions as src

# import data
adata = anndata.read_h5ad("proc/adata_filtered.h5ad")
cpm = anndata.read_h5ad("/scratch/s/shreejoy/nxu/AIBS/patch-seq/transcriptomics/20200513_Mouse_PatchSeq_Release_cpm.v2.h5ad")
count_matrix = pd.read_csv("data/20200513_Mouse_PatchSeq_Release_count.v2.csv", index_col=0)

# get mappings
metadata = pd.read_csv("data/20200711_patchseq_metadata_mouse.csv")
sample_id_to_cell_id = metadata.set_index("transcriptomics_sample_id")["cell_specimen_id"].to_dict()    
file_manifest = pd.read_csv("data/2021-09-13_mouse_file_manifest.csv").query("file_type == 'forward_fastq'")
file_manifest["file_name"] = file_manifest["file_name"].str.removesuffix("_R1.fastq.gz")
cell_id_to_file_name = file_manifest.set_index("cell_specimen_id")["file_name"].to_dict()

# map transcriptomic_sample_id to file_name
cpm.obs.index = cpm.obs.index.map(sample_id_to_cell_id).map(cell_id_to_file_name)
cpm = cpm[adata.obs.index, :]

# write count_adata for use in monocle3
count_matrix.columns = count_matrix.columns.map(sample_id_to_cell_id).map(cell_id_to_file_name)
count_adata = anndata.AnnData(csr_matrix(count_matrix.T.values))
count_adata.obs.index = count_matrix.columns
count_adata.var.index = count_matrix.index
count_adata = src.add_ephys_prop(count_adata)
count_adata.write("proc/adata_count.h5ad")

# align obs
cpm = cpm.to_df()
cpm = cpm.loc[adata.obs.index]

# def run_regression_per_intron_group(intron_group_name):
#     gene_name = intron_group_name.split("_")[0]
#     df_intron_group, psi = ds.run_regression(adata.X, cpm[gene_name], adata.var, intron_group_name)
#     return df_intron_group

# reload(src)
adata = adata[:, adata.var.gene_name.isin(cpm.columns)]

ctx = mp.get_context('spawn')
with ctx.Pool(40) as pool:
    run_regression_per_intron_group_partial = partial(ds.run_regression_per_intron_group, adata, cpm)
    results = pool.map_async(run_regression_per_intron_group_partial, 
                             adata.var.intron_group.unique().to_numpy(), chunksize=4000)
    results.wait()
    results = results.get()