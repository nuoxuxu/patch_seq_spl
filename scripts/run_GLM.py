import scanpy as sc
import scquint.differential_splicing as ds
import pandas as pd
from patch_seq_spl.helper_functions import add_predictors

adata = sc.read_h5ad(snakemake.input[0])
adata = add_predictors(adata)

if snakemake.wildcards.predictor == "cpm":
    df_list = []
    for intron_group in adata.var.intron_group.unique().to_numpy():
        gene_name = intron_group.split("_")[0]
        if snakemake.wildcards.model == "simple":
            reduced = "1"
            full = f"{gene_name}"
        if snakemake.wildcards.model == "multiple":
            reduced = "subclass"
            full = f"subclass + {gene_name}"   
        df, _ = ds.run_regression(adata, intron_group, reduced, full)
        df_list.append(df)
else:
    if snakemake.wildcards.model == "simple":
        reduced = "1"
        full = f"{snakemake.wildcards.predictor}"

    if snakemake.wildcards.model == "multiple":
        reduced = "subclass"
        full = f"subclass + {snakemake.wildcards.predictor}"

    df_list = []
    for intron_group in adata.var.intron_group.unique().to_numpy():
        df, _ = ds.run_regression(adata, intron_group, reduced, full)
        df_list.append(df)

    df = pd.concat(df_list)

df.to_csv(snakemake.output[0])