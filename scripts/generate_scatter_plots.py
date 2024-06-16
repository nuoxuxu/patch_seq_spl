import os
from pathlib import Path
import pandas as pd
import numpy as np
from src.extended_anndata import *
import anndata
from src.df_accessors import *

adata = anndata.read_h5ad("proc/scquint/preprocessed_adata_three.h5ad")
adata = ExtendedAnnData(adata)
VGIC_LGIC = np.load("data/VGIC_LGIC.npy", allow_pickle=True)
adata = adata[:, adata.var.gene_name.isin(VGIC_LGIC)]
adata = adata.add_predictors()

glm_results = adata.uns["simple"]
sig_VGIC_SJ = glm_results.glm.rank_introns_by_n_sig_corr(rank_by="all", VGIC_only=True, sig_only=True)

for intron_group in sig_VGIC_SJ.index:
    adata.save_scatter_plots_per_intron_group(adata, intron_group, "rheobase_i", True)

for intron_group in sig_VGIC_SJ.index:
    adata.save_scatter_plots_per_intron_group(adata, intron_group, "rheo_upstroke_downstroke_ratio", True)    

for intron_group in sig_VGIC_SJ.index:
    adata.save_scatter_plots_per_intron_group(adata, intron_group, "rheo_width", True)