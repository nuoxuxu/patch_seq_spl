#!/usr/bin/env python3
import polars as pl
import argparse

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True, help="Name of the Patch-seq dataset")
    parser.add_argument("--metadata", required=True, help="Metadata")
    parser.add_argument("--file_manifest", required=True, help="File manifest")
    args = parser.parse_args()

    file_manifest = pl.read_csv(args.file_manifest)
    metadata = pl.read_csv(args.metadata)

    cell_to_sample = file_manifest\
        .filter(pl.col("file_type")=="fastq")\
        .with_columns(
            pl.col("file_name").str.replace(".fastq.tar", "", literal=True)
        )\
        .rename({"file_name":"sample_name", "specimen_id": "cell_specimen_id"})\
        .select(["cell_specimen_id", "sample_name"])

    sample_sheet = metadata\
        .select(["Cell ID", "MET-type"])\
        .rename({"Cell ID":"cell_specimen_id", "MET-type":"cell_type_label"})\
        .join(cell_to_sample, on="cell_specimen_id", how="inner")\
        .with_columns(
            pl.col("sample_name").map_elements(lambda x: f"{args.dataset}/{x}.SJ.out.tab", return_dtype=pl.String).alias("path")
        )\
        .drop("sample_name")

    sample_sheet.write_csv(f"proc/sample_sheet_{args.dataset}.csv")

if __name__ == "__main__":
    main()