import polars as pl
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from src2.utils import correlate
from sklearn.impute import SimpleImputer
from statsmodels.stats.multitest import fdrcorrection

ratio_matrix = pl.read_csv("lampert_data/ratio_matrix.csv")
ephys_props = pl.read_csv("lampert_data/ephys_prop.csv")
metadata = pl.read_csv("lampert_data/patchseq-meta.csv", null_values = "NA")
SJ_pig_genes = pl.read_csv("lampert_data/SJ_pig_genes.csv")

def impute(df):
    # df = your Polars DataFrame
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns

    # 1 Convert the numeric slice to a NumPy array
    X = df.select(numeric_cols).to_numpy()      # shape (n_rows, n_cols)

    # 2 Impute with mean
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)        # still a NumPy array

    # 3 Push the imputed block back into Polars
    df_filled = df.with_columns(
        pl.DataFrame(X_imputed, schema=numeric_cols)   # overwrites those columns
    )
    return df_filled

# Getting cell-type-level ephys props

ephys_props_pb = ephys_props\
    .join(
        metadata["cell_id", "labels"], 
        on = "cell_id",
        how = "left"
    )\
    .drop("cell_id")\
    .group_by("labels")\
    .mean()\
    .sort("labels")

# Getting cell-type-level ratio matrix

combined = pl.scan_csv(
        list(Path("lampert_data/tab_files").iterdir()), 
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
    ).drop("path")\
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
    )["SJ_group"]

sharing_end = sharing_end.filter(pl.col("SJ_group").is_in(SJ_group_to_keep))

# Aggregating SJ reads in the same cell type

sharing_end = sharing_end\
    .join(
        metadata["cell_id", "labels"], 
        on = "cell_id",
        how = "left"
    )\
    .drop("cell_id")\
    .group_by("labels", "SJ")\
    .agg(
        pl.col("unique").sum(),
        pl.col("SJ_group").first()
    )\
    .pivot(index="SJ", on="labels", values="unique")\
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
    .drop("pig_gene_name")\
    .with_columns(
        SJ_group = pl.col("SJ").str.replace(r"_[^_]*$", "")
    )\
    .fill_null(0)

# Getting the SJ ratio matrix

total_unique_per_group = sharing_end\
    .drop("SJ")\
    .group_by("SJ_group")\
    .sum()

denominator = sharing_end\
    .join(
        total_unique_per_group, 
        on="SJ_group", 
        how="left"
    )\
    .select(
        pl.selectors.contains("_right")
    )

numerator = sharing_end\
    .join(
        total_unique_per_group, 
        on="SJ_group", 
        how="left"
    )\
    .select(
        pl.selectors.exclude(pl.selectors.contains("_right"))
    ).drop("SJ_group", "SJ")

ratio_matrix = numerator / denominator
ratio_matrix = pl.concat([sharing_end.select("SJ"), ratio_matrix], how="horizontal")
ratio_matrix = ratio_matrix.drop("SJ").transpose(include_header=True, column_names=ratio_matrix["SJ"], header_name="labels")
ratio_matrix = ratio_matrix.sort("labels", descending=False)

ratio_matrix\
    .select(
        pl.selectors.contains("labels"),
        pl.selectors.contains("SCN")
    ).write_csv("lampert_data/ratio_matrix_pb.csv")

ephys_props_pb.write_csv("lampert_data/ephys_props_pb.csv")

ratio_matrix_pb["SCN9A_15:72912040_-_72846418"]
ephys_props_pb["PC1"]

plt.plot(ratio_matrix_pb["SCN9A_15:72912040_-_72846418"], ephys_props_pb["PC1"], "o")
plt.savefig("lampert_data/SCN9A_PC1.png")