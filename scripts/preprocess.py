import scanpy as sc
import pandas as pd
from src.extended_anndata import *

adata = sc.read_h5ad(snakemake.input.adata_path)
adata = ExtendedAnnData(adata)

adata = adata.filter_adata((snakemake.config["min_global_SJ_counts"], snakemake.config["min_cells_per_feature"], snakemake.config["min_cells_per_intron_group"]))
adata = adata.update_intron_group_size()

adata.var.annotation = ((adata.var.gene_id_start != "") & (adata.var.gene_id_end != "")).astype(int)

adata.write(snakemake.output[0])