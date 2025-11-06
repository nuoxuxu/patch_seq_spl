#!/usr/bin/env python3
import polars as pl
from pathlib import Path
import os
import argparse

def get_combined(sample_sheet):
    all_files = [Path(SJ_file) for SJ_file in sample_sheet["path"]]
    non_empty_files = [f for f in all_files if os.path.getsize(f) > 0]

    combined = pl.scan_csv(
            non_empty_files,
            separator="\t", has_header= False, 
            new_columns=["chrom", "start", "end", "strand", "motif", "annotated", "unique_reads", "multi_reads", "max_overhang"],
            schema_overrides={"chrom": pl.String},
            include_file_paths = "path"
        ).collect()\
        .filter(
            pl.col("unique_reads")!=0
        )\
        .group_by("chrom", "start", "end", "strand", "path")\
        .agg(
            pl.col("motif").first(),
            pl.col("annotated").first(),
            unique=pl.col("unique_reads").sum(),
            multi=pl.col("multi_reads").first(),
            max_overhang=pl.col("max_overhang").first(),
        )\
        .select("path", pl.selectors.exclude("path"))\
        .join(
            sample_sheet,
            on="path",
            how="left"
        )\
        .drop("path")\
        .select("cell_specimen_id", "cell_type_label", pl.selectors.exclude("cell_specimen_id", "cell_type_label"))\
        .with_columns(
            pl.col("start").cast(pl.String),
            pl.col("end").cast(pl.String),
            pl.col("strand").cast(pl.String)
        )\
        .with_columns(
            pl.col("strand").replace({1: "+", 2: "-"})
        )
    return combined

def filter_uncannonical_introns(combined):
    combined = combined.filter(
        ~((pl.col("annotated") == 0) & (pl.col("motif") != 0) & (pl.col("max_overhang") < 20))
    ).filter(
        ~((pl.col("annotated") == 0) & (pl.col("motif") == 0) & (pl.col("max_overhang") < 30))
    )
    return combined

def add_SJ_and_SJ_group_label(combined, sharing):
    if sharing == "three":
        out = combined.with_columns(
            SJ = pl.when(pl.col("strand") == "+")\
                .then(pl.col("gene_name") + pl.lit("_") + pl.col("chrom") + pl.lit(":") + pl.col("end") + pl.lit("_") + pl.col("strand") + pl.lit("_") + pl.col("start"))\
                .otherwise(pl.col("gene_name") + pl.lit("_") + pl.col("chrom") + pl.lit(":") + pl.col("start") + pl.lit("_") + pl.col("strand") + pl.lit("_") + pl.col("end")),
            SJ_group = pl.when(pl.col("strand") == "+")\
                .then(pl.col("gene_name") + pl.lit("_") + pl.col("chrom") + pl.lit(":") + pl.col("end") + pl.lit("_") + pl.col("strand"))\
                .otherwise(pl.col("gene_name") + pl.lit("_") + pl.col("chrom") + pl.lit(":") + pl.col("start") + pl.lit("_") + pl.col("strand"))
            ).drop("gene_name")
    elif sharing == "five":
        out = combined.with_columns(
            SJ = pl.when(pl.col("strand") == "+")\
                .then(pl.col("gene_name") + pl.lit("_") + pl.col("chrom") + pl.lit(":") + pl.col("start") + pl.lit("_") + pl.col("strand") + pl.lit("_") + pl.col("end"))\
                .otherwise(pl.col("gene_name") + pl.lit("_") + pl.col("chrom") + pl.lit(":") + pl.col("end") + pl.lit("_") + pl.col("strand") + pl.lit("_") + pl.col("start")),
            SJ_group = pl.when(pl.col("strand") == "+")\
                .then(pl.col("gene_name") + pl.lit("_") + pl.col("chrom") + pl.lit(":") + pl.col("start") + pl.lit("_") + pl.col("strand"))\
                .otherwise(pl.col("gene_name") + pl.lit("_") + pl.col("chrom") + pl.lit(":") + pl.col("end") + pl.lit("_") + pl.col("strand"))      
            ).drop("gene_name")
    else:
        raise ValueError(f"Unknown sharing mode: {sharing}. Expected 'three' or 'five'.")
    return out

def filter_unique_SJ_per_cell(df, min_unique_SJ=10000, ):
    original_cell_count = df.unique("cell_specimen_id").shape[0]
    cells_to_keep = df.group_by("cell_specimen_id")\
        .agg(pl.col("SJ").len())\
        .filter(pl.col("SJ") >= min_unique_SJ)["cell_specimen_id"].to_list()
    removed_cell_count = original_cell_count - len(cells_to_keep)
    print(f"Filtering cells with less than {min_unique_SJ} unique SJ: {removed_cell_count} cells removed from {original_cell_count} cells.")
    return df.filter(pl.col("cell_specimen_id").is_in(cells_to_keep))

def filter_SJ_group_based_on_min_cells(df, min_cells=15):
    SJ_group_to_keep = df.group_by("SJ_group")\
        .agg(
            pl.col("cell_specimen_id").count()
        )\
        .filter(
            pl.col("cell_specimen_id") > min_cells
        )["SJ_group"].to_list()

    df = df.filter(pl.col("SJ_group").is_in(SJ_group_to_keep))
    return df

def filter_min_total_SJ_per_cell(df, min_global_counts=50000):
    original_cell_count = df.unique("cell_specimen_id").shape[0]
    cells_to_keep = df.group_by("cell_specimen_id")\
        .agg(
            pl.col("unique").sum()
        )\
        .filter(
            pl.col("unique") >= min_global_counts
        )["cell_specimen_id"].to_list()
    removed_cell_count = original_cell_count - len(cells_to_keep)
    print(f"Filtering cell with less than {min_global_counts} global unique counts: {removed_cell_count} cell removed from {original_cell_count} cell.")
    df = df.filter(pl.col("cell_specimen_id").is_in(cells_to_keep))
    return df

def filter_total_unique_per_SJ_group(df, min_total_unique_per_SJ_group=30):
    return df\
        .with_columns(
            pl.col("unique").sum().over("SJ_group").alias("total_unique_per_SJ_group")
        )\
        .filter(
            pl.col("total_unique_per_SJ_group") > min_total_unique_per_SJ_group
        )

def get_ratio_matrix(df, mode):
    if mode == "cell":
        index_col = "cell_specimen_id"
    elif mode == "ttype":
        index_col = "cell_type_label"
        # Perform the initial aggregation required for 'ttype' mode
        df = df.group_by(index_col, "SJ").agg(
            pl.col("chrom").first(),
            pl.col("start").first(),
            pl.col("end").first(),
            pl.col("strand").first(),
            pl.col("unique").sum(),
            pl.col("SJ_group").first()
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'cell' or 'ttype'.")

    ratio_matrix = df\
        .with_columns(
            num_SJ_for_cell = pl.col("SJ").len().over("SJ_group", index_col)
        )\
        .with_columns(
            pl.col("num_SJ_for_cell").max().over("SJ_group")
        )\
        .filter(
            pl.col("num_SJ_for_cell") > 1
        )\
        .drop("num_SJ_for_cell")\
        .with_columns(
            ratio = pl.col("unique") / pl.col("unique").sum().over("SJ_group", index_col)
        ).select(
            index_col, "SJ", "ratio"
        ).pivot(
            index=index_col,
            on="SJ",
            values="ratio"
        )\
        .with_columns(
            pl.col(index_col).cast(pl.String)
        )

    return ratio_matrix

def preprocess_ephys_data(file_manifest, metadata, ephys_data, mode, min_cell_per_ttype=15):
    file_manifest = pl.read_csv(file_manifest, schema_overrides={"cell_specimen_id": pl.String})
    metadata = pl.read_csv(metadata, schema_overrides={"cell_specimen_id": pl.String})
    name_to_cell_specimen_id = file_manifest\
        .filter(pl.col("file_type")=="forward_fastq")\
        .with_columns(
            pl.col("file_name").str.replace("_R1.fastq.gz", "")
        )\
        .select("file_name", "cell_specimen_id")
    ephys_data = pl.read_csv(ephys_data)\
        .rename({"cell_specimen_id": "file_name"})\
        .join(
            name_to_cell_specimen_id,
            on="file_name", how="left"
        ).drop("file_name")\
        .join(
            metadata["cell_specimen_id", "T-type Label"],
            on="cell_specimen_id", how="left"
        )\
        .with_columns(
            subclass = pl.col("T-type Label").str.split(" ").map_elements(lambda x:x[0], return_dtype=pl.String)
        )
    if mode == "cell":
        ephys_data = ephys_data\
            .drop("T-type Label")\
            .select("cell_specimen_id", "subclass", pl.selectors.exclude("cell_specimen_id", "subclass"))
    elif mode == "ttype":
        ephys_data = ephys_data\
            .with_columns(
                pl.col("cell_specimen_id").len().over("T-type Label").alias("n_cells")
            )\
            .filter(pl.col("n_cells") >= min_cell_per_ttype)\
            .drop("n_cells", "cell_specimen_id")\
            .group_by("T-type Label")\
            .agg(
                pl.selectors.exclude("subclass").mean(),
                subclass = pl.col("subclass").first()
            )
        ephys_data = ephys_data.rename({"T-type Label": "cell_type_label"})
    return ephys_data

def read_gtf(file, attributes=["transcript_id"], keep_attributes=True):
    if keep_attributes:
        return pl.read_csv(file, separator="\t", comment_prefix="#", schema_overrides = {"seqname": pl.String}, has_header = False, new_columns=["seqname","source","feature","start","end","score","strand","frame","attributes"])\
            .with_columns(
                [pl.col("attributes").str.extract(rf'{attribute} "([^;]*)";').alias(attribute) for attribute in attributes]
                )
    else:
        return pl.read_csv(file, separator="\t", comment_prefix="#", schema_overrides = {"seqname": pl.String}, has_header = False, new_columns=["seqname","source","feature","start","end","score","strand","frame","attributes"])\
            .with_columns(
                [pl.col("attributes").str.extract(rf'{attribute} "([^;]*)";').alias(attribute) for attribute in attributes]
                ).drop("attributes")

def gtf_to_SJ(gtf):
    import numpy as np
    out = gtf\
        .filter(
            pl.col("feature")=="exon"
        )\
        .group_by("transcript_id", maintain_order=True)\
        .agg(
            pl.col("strand").first(),
            pl.col("seqname").first(),
            pl.col("start"),
            pl.col("end"),
            pl.col("attributes").len(),
            pl.col("gene_name").first(),
            pl.col("gene_id").first()
        )\
        .filter(pl.col("attributes")>1)\
        .with_columns(
            pl.col("start").map_elements(lambda l: np.sort(np.array(l)-1)[1:].tolist(), return_dtype=pl.List(pl.Int64)).alias("end"),
            pl.col("end").map_elements(lambda l: np.sort(np.array(l)+1)[:-1].tolist(), return_dtype=pl.List(pl.Int64)).alias("start")
        )\
        .explode("start", "end")\
        .rename({"seqname": "chrom"})\
        .filter(pl.col("start").is_null().not_())
    return out

def add_gene_annotation(combined, gtf_file):
    ensembl_SJ = read_gtf(gtf_file, attributes=["gene_name", "gene_id", "transcript_id"])\
        .pipe(gtf_to_SJ)\
        .with_columns(
            pl.col("start").cast(pl.String),
            pl.col("end").cast(pl.String),
            gene_name = pl.when(pl.col("gene_name").is_null()).then(pl.col("gene_id")).otherwise(pl.col("gene_name"))
        )\
        .select("chrom", "start", "end", "strand", "gene_name")\
        .unique(["chrom", "start", "end", "strand"])

    ensembl_genes = read_gtf(gtf_file, attributes=["gene_name", "gene_id"])\
        .filter(pl.col("feature")=="gene")\
        .select("seqname", "gene_name", "gene_id", "start", "end", "strand")\
        .rename({"seqname":"chrom", "start":"gene_start", "end":"gene_end"})\
        .with_columns(
            gene_name = pl.when(pl.col("gene_name").is_null()).then(pl.col("gene_id")).otherwise(pl.col("gene_name"))
        )\
        .drop("gene_id")\
        .unique("gene_name")

    combined_with_gene_name = combined.join(
        ensembl_SJ,
        on=["chrom", "start", "end", "strand"],
        how="left"
    )

    known_SJ = combined_with_gene_name.filter(pl.col("gene_name").is_not_null())\
        .unique(["cell_specimen_id", "chrom", "start", "end", "strand"])

    novel_SJ = combined_with_gene_name\
        .filter(pl.col("gene_name").is_null())\
        .with_columns(
            pl.col("start").cast(pl.Int64),
            pl.col("end").cast(pl.Int64)
        )\
        .join(
            ensembl_genes,
            on=["chrom", "strand"],
            how="left"
        )\
        .filter(
            (pl.col("start").cast(pl.Int64) >= pl.col("gene_start")) &
            (pl.col("end").cast(pl.Int64) <= pl.col("gene_end"))
        )\
        .drop("gene_name")\
        .rename({"gene_name_right":"gene_name"})\
        .drop(["gene_start", "gene_end"])\
        .with_columns(
            pl.col("start").cast(pl.String),
            pl.col("end").cast(pl.String)
        )\
        .unique(["cell_specimen_id", "chrom", "start", "end", "strand"])

    return pl.concat([known_SJ, novel_SJ])

def align_data(ratio_matrix, ephys_data):
    index_name = ratio_matrix.columns[0]
    shared_names = pl.DataFrame(list(set(ratio_matrix[:,0]).intersection(set(ephys_data[:,0])))).rename({"column_0": index_name})
    ratio_matrix = shared_names\
        .join(ratio_matrix, on=index_name, how="left")
    ephys_data = shared_names\
        .join(ephys_data, on=index_name, how="left")
    return ratio_matrix, ephys_data

def main():
    parser = argparse.ArgumentParser(description="Preprocess splicing junction data.")
    parser.add_argument("--sample_sheet", type=str, required=True, help="Path to the sample sheet CSV file.")
    parser.add_argument("--file_manifest", type=str, required=True, help="Path to the file manifest CSV file.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--ephys_data", type=str, required=True, help="Path to the ephys data CSV file.")
    parser.add_argument("--gtf_file", type=str, required=True, help="Path to the GTF annotation file.")
    parser.add_argument("--sharing", type=str, required=True, help="Sharing mode for SJ grouping ('end' or 'start').")
    parser.add_argument("--mode", type=str, required=True, help="Mode for processing ('cell' or 'ttype').")
    args = parser.parse_args()

    sample_sheet = pl.read_csv(args.sample_sheet, schema_overrides={"cell_specimen_id": pl.String})
    combined = get_combined(sample_sheet)
    combined_uncanonnical_filtered = filter_uncannonical_introns(combined)
    combined_uncanonnical_filtered = add_gene_annotation(combined_uncanonnical_filtered, args.gtf_file)
    combined_uncanonnical_filtered = add_SJ_and_SJ_group_label(combined_uncanonnical_filtered, sharing=args.sharing)
    combined_uncanonnical_filtered = filter_unique_SJ_per_cell(combined_uncanonnical_filtered, 10000)
    combined_uncanonnical_filtered = filter_min_total_SJ_per_cell(combined_uncanonnical_filtered)
    combined_uncanonnical_filtered = filter_total_unique_per_SJ_group(combined_uncanonnical_filtered)

    ratio_matrix = get_ratio_matrix(combined_uncanonnical_filtered, mode=args.mode)\
        .fill_null(0.0)
    
    ephys_data = preprocess_ephys_data(
        file_manifest=args.file_manifest,
        metadata=args.metadata,
        ephys_data=args.ephys_data,
        mode=args.mode
    )

    ratio_matrix, ephys_data = align_data(ratio_matrix, ephys_data)

    ratio_matrix.write_parquet(f"proc/ratio_matrix_{args.sharing}_{args.mode}.parquet")
    ephys_data.write_parquet(f"proc/ephys_data_{args.sharing}_{args.mode}.parquet")

if __name__ == "__main__":
    main()

# sample_sheet = pl.read_csv("proc/sample_sheet.csv")
# combined = get_combined(sample_sheet)
# combined_uncanonnical_filtered = filter_uncannonical_introns(combined)
# combined_uncanonnical_filtered = add_gene_annotation(combined_uncanonnical_filtered, Path(os.environ["GENOMIC_DATA_DIR"]).joinpath("Ensembl/Mouse/Release_110/Raw/Mus_musculus.GRCm39.110.gtf"))
# combined_uncanonnical_filtered = add_SJ_and_SJ_group_label(combined_uncanonnical_filtered, sharing="three")
# combined_uncanonnical_filtered = filter_unique_SJ_per_cell(combined_uncanonnical_filtered, 10000)
# combined_uncanonnical_filtered = filter_min_total_SJ_per_cell(combined_uncanonnical_filtered)
# combined_uncanonnical_filtered = filter_total_unique_per_SJ_group(combined_uncanonnical_filtered)