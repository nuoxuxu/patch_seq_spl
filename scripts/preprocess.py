from patch_seq_spl.helper_functions import filter_adata, update_intron_group_size
import scanpy as sc
import pandas as pd

# filter

adata = sc.read_h5ad(snakemake.input.adata_path)
adata = filter_adata(
    adata,
    (snakemake.config["min_global_SJ_counts"], snakemake.config["min_cells_per_feature"], snakemake.config["min_cells_per_intron_group"]))

adata = update_intron_group_size(adata)
adata.write(snakemake.output[0])