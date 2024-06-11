from patch_seq_spl.helper_functions import filter_adata, update_intron_group_size
import scanpy as sc
import pandas as pd
import patch_seq_spl.helper_functions as src


adata = sc.read_h5ad(snakemake.input.adata_path)
adata = src.ExtendedAnnData(adata)

adata = adata.filter_adata((snakemake.config["min_global_SJ_counts"], snakemake.config["min_cells_per_feature"], snakemake.config["min_cells_per_intron_group"]))
adata = adata.update_intron_group_size()

adata.var.annotation = ((adata.var.gene_id_start != "") & (adata.var.gene_id_end != "")).astype(int)

adata.write(snakemake.output[0])