import os
from pathlib import Path
import pandas as pd
import numpy as np
import patch_seq_spl.helper_functions as src
import anndata

adata = anndata.read_h5ad("proc/scquint/preprocessed_adata_three.h5ad")
VGIC_LGIC = np.load("data/VGIC_LGIC.npy", allow_pickle=True)
adata = adata[:, adata.var.gene_name.isin(VGIC_LGIC)]

glm_results = src.get_glm_results("proc/scquint/three/simple")
glm_results = src.rank_introns_by_n_sig_corr(glm_results, "all")
gene_names = src.get_gene_from_intron_group(glm_results.index)
sig_VGIC_SJ = glm_results.iloc[src.get_VGIC_idx(gene_names)]
adata = src.add_predictors(adata)

for intron_group in sig_VGIC_SJ.index:
    src.save_scatter_plots_per_intron_group(adata, intron_group, "rheobase_i", True)