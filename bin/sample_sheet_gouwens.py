#!/usr/bin/env python3
import polars as pl
import argparse

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True, help="Name of the Patch-seq dataset")
    parser.add_argument("--metadata", required=True, help="Metadata")
    parser.add_argument("--file_manifest", required=True, help="File manifest")
    args = parser.parse_args()

    metadata = pl.read_csv(args.metadata, null_values = "NA")
    file_manifest = pl.read_csv(args.file_manifest)

    cell_to_sample = file_manifest\
        .filter(pl.col("file_type")=="forward_fastq")\
        .with_columns(
            pl.col("file_name").str.replace("_R1.fastq.gz", "", literal=True)
        )\
        .rename({"file_name":"sample_name"})\
        .select(["cell_specimen_id", "sample_name"])

    sample_sheet = metadata\
        .select(["cell_specimen_id", "T-type Label"])\
        .join(cell_to_sample, on="cell_specimen_id", how="left")\
        .with_columns(
            pl.col("sample_name").map_elements(lambda x: f"{args.dataset}/{x}.SJ.out.tab", return_dtype=pl.String).alias("path")
        )\
        .drop("sample_name")
    
    sample_sheet = sample_sheet\
        .filter(pl.col("path").str.contains("SM-GBMG3_E1-50_AGTTAGCTGG-CATTCTCATC").not_())

    sample_sheet = sample_sheet.rename({"T-type Label":"cell_type_label"})
    
    sample_sheet.write_csv(f"sample_sheet_{args.dataset}.csv")

if __name__ == "__main__":
    main()