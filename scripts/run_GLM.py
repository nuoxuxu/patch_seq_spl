import scanpy as sc
import src.differential_splicing as ds
from src.extended_anndata import *
import pandas as pd

#TODO move run_regression to extended_anndata
adata = sc.read_h5ad(snakemake.input[0])
adata = ExtendedAnnData(adata)
adata = adata.add_predictors()

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