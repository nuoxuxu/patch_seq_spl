#!/usr/bin/env python3
import polars as pl
from pathlib import Path
from src2 import utils
import numpy as np
import pandas as pd
import polars.selectors as cs
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection

input_dir = "lampert_data/tab_files"
gtf_path = "/project/s/shreejoy/Genomic_references/Ensembl/Homo_sapiens.GRCh38.113.chr.gtf"
path_to_metadata = "lampert_data/patchseq-meta.csv"

metadata = pl.read_csv(path_to_metadata, null_values = "NA")

combined = pl.scan_csv(
        list(Path(input_dir).iterdir()), 
        separator="\t", has_header= False, 
        new_columns=["chrom", "start", "end", "strand", "motif", "annotated", "unique_reads", "multi_reads", "max_overhang"],
        schema_overrides={"chrom": pl.String},
        include_file_paths = "path"
    ).collect()\
    .with_columns(
        cell_id = pl.col("path").str.split("_").map_elements(lambda x: x[2], return_dtype=pl.String)
    )\
    .with_columns(
        pl.col("cell_id").str.split("/").map_elements(lambda x: x[1], return_dtype=pl.String)
    ).drop("path")

combined = combined\
    .filter(
        pl.col("unique_reads")!=0
    )\
    .group_by("chrom", "start", "end", "strand", "cell_id")\
    .agg(
        pl.col("motif").first(),
        pl.col("annotated").first(),
        unique=pl.col("unique_reads").sum(),
        multi=pl.col("multi_reads").first(),
        max_overhang=pl.col("max_overhang").first(),
    )

# filter out some uncannonical introns
combined = combined.filter(
    ~((pl.col("annotated") == 0) & (pl.col("motif") != 0) & (pl.col("max_overhang") < 20))
).filter(
    ~((pl.col("annotated") == 0) & (pl.col("motif") == 0) & (pl.col("max_overhang") < 30))
)

# Sharing end for now
sharing_end = combined\
    .with_columns(
        pl.col("start").cast(pl.String),
        pl.col("end").cast(pl.String),
        pl.col("strand").cast(pl.String),
        pl.col("cell_id").cast(pl.Int64),
    )\
    .with_columns(
        pl.col("strand").replace({1: "+", 2: "-"}),
    )\
    .with_columns(
        SJ_group = pl.col("chrom") + pl.lit(":") + pl.col("end") + pl.lit("_") + pl.col("strand"),
        SJ = pl.col("chrom") + pl.lit(":") + pl.col("end") + pl.lit("_") + pl.col("strand") + pl.lit("_") + pl.col("start"),
    )\
    .filter(
        pl.col("cell_id").is_in(metadata["cell_id"].to_list())
    )

# Keeping only SJ_group that are in more than 15 cells
SJ_group_to_keep = sharing_end.group_by("SJ_group")\
    .agg(
        pl.col("cell_id").count()
    )\
    .filter(
        pl.col("cell_id") > 15
    ).select("SJ_group")

sharing_end = sharing_end.filter(pl.col("SJ_group").is_in(SJ_group_to_keep))

# Getting the SJ ratio matrix
total_unique_per_group = sharing_end.group_by("SJ_group").agg(pl.col("unique").sum()).rename({"unique": "total_unique_per_group"})

sharing_end = sharing_end\
    .join(
        total_unique_per_group, 
        on="SJ_group", 
        how="left"
    )\
    .with_columns(
        ratio = pl.col("unique") / pl.col("total_unique_per_group")
    )

ratio_matrix = sharing_end.select("SJ", "ratio", "cell_id")\
    .pivot(
        index="cell_id",
        on="SJ",
        values="ratio"
    )\
    .sort("cell_id")\
    .drop("cell_id")\
    .fill_null(strategy="mean")

# Getting the ephys props
ephys_prop = metadata['RMP', 'pA_start', 'pA_threshold', 'Rheobase', 'input_res', 'Input_res_new', 'AP_threshhold', 'AP_max', 'Apamp', 'Min_AHP', 'Loc_APHW', 'APD', 'max_slope_raw', 'max_slope_smth', 'slope_oversh', 'PC1']\
    .fill_null(strategy="mean")

# Correlating the ratio matrix with the ephys properties
corr_matrix, p_value_matrix = utils.correlate(ratio_matrix, ephys_prop)

p_value_matrix = pl.concat([pl.Series("SJ", ratio_matrix.columns).to_frame(), p_value_matrix], how="horizontal")

p_value_matrix = p_value_matrix.drop_nans()

_, qvals = fdrcorrection(p_value_matrix.drop("SJ").to_numpy().flatten())

corr_p_value_matrix = pd.DataFrame(
    qvals.reshape( p_value_matrix.shape[0], p_value_matrix.shape[1] - 1 ),
    index = p_value_matrix["SJ"],
    columns = p_value_matrix.drop("SJ").columns
)\
.pipe(pl.from_pandas, include_index=True)\
.rename({"None": "SJ"})\
.with_columns(
    cs.exclude("SJ") < 0.05
)\
.with_columns(
    sum = pl.sum_horizontal(cs.exclude("SJ"))
)\
.filter(pl.col("sum") != 0)

# Add back the gene names
corr_p_value_matrix.select("SJ").write_csv("lampert_data/SJ_names.csv")
SJ_pig_genes = pl.read_csv("lampert_data/SJ_pig_genes.csv")

corr_p_value_matrix = corr_p_value_matrix\
    .join(
        SJ_pig_genes,
        on="SJ",
        how="left"
    )\
    .filter(
        pl.col("pig_gene_name").is_not_null()
    )\
    .with_columns(
        SJ = pl.col("pig_gene_name") + pl.lit("_") + pl.col("SJ")
    )\
    .drop("pig_gene_name")

# Plotting

SJ_to_show = p_value_matrix\
    .with_columns(
        cs.exclude("SJ") < 0.05
    )\
    .with_columns(
        sum = pl.sum_horizontal(cs.exclude("SJ"))
    )\
    .sort("sum", descending=True)[:50, :]\
    .select("SJ")

df = p_value_matrix.filter(pl.col("SJ").is_in(SJ_to_show))

df = pd.DataFrame(
    -np.log10(df.drop("SJ").to_numpy()),
    index = df["SJ"],
    columns = df.drop("SJ").columns
)

fig, ax = plt.subplots(figsize=(14, 10))

sns.heatmap(
    data = df,
    ax = ax
)

plt.savefig("heatmap.pdf", bbox_inches='tight')

# Looking at the genes
SJ_to_show\
    .with_columns(
        gene_name = pl.col("SJ").str.split("_").map_elements(lambda x: x[0], return_dtype=pl.String)
    )\
    .select("gene_name")\
    .write_csv("lampert_data/genes_to_check.csv")


corr_p_value_matrix.sort("sum", descending=True).write_csv("lampert_data/corr_p_value_matrix.csv")