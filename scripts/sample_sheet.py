#!/usr/bin/env python3
import polars as pl
def main():

    metadata = pl.read_csv("data/20200711_patchseq_metadata_mouse.csv", null_values = "NA")
    file_manifest = pl.read_csv("data/2021-09-13_mouse_file_manifest.csv")

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
            pl.col("sample_name").map_elements(lambda x: f"proc/star/{x}.SJ.out.tab", return_dtype=pl.String).alias("path")
        )\
        .drop("sample_name")
    
    sample_sheet = sample_sheet\
        .filter(pl.col("path").str.contains("SM-GBMG3_E1-50_AGTTAGCTGG-CATTCTCATC").not_())

    sample_sheet = sample_sheet.rename({"T-type Label":"cell_type_label"})
    
    sample_sheet.write_csv("proc/sample_sheet.csv")

if __name__ == "__main__":
    main()