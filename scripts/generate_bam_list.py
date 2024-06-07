import json
import pandas as pd
import random
from pathlib import Path

metadata = pd.read_csv('data/20200711_patchseq_metadata_mouse.csv')
with open("data/mappings/transcriptomics_sample_id_file_name.json", "r") as f:
    transcriptomics_sample_id_file_name = json.load(f)
metadata["filename"] = metadata.transcriptomics_sample_id.map(transcriptomics_sample_id_file_name)
metadata.dropna(subset=["filename"], inplace=True)
metadata["full_path"] = metadata["filename"].apply(lambda x: "".join(["/external/rprshnas01/netdata_kcni/stlab/Nuo/STAR_for_SGSeq/coord_bams/", x, "Aligned.sortedByCoord.out.bam"]) if x else None)
metadata["T-type Label"] = metadata["T-type Label"].map(lambda x: "_".join(x.split(" ")))
my_dict = metadata.groupby("T-type Label")["full_path"].apply(lambda x: x.tolist()).to_dict()

file_list = my_dict[Path(snakemake.output[0]).stem]
if len(file_list) > 20:
    if len(file_list) > 50:
        file_list = random.sample(file_list, 50)
    with open(snakemake.output[0], "w") as f:
        for file in file_list:
            f.write(file + "\n")