import pandas as pd
import json
import os

metadata = pd.read_csv('data/20200711_patchseq_metadata_mouse.csv')
with open("data/mappings/transcriptomics_sample_id_file_name.json", "r") as f:
    transcriptomics_sample_id_file_name = json.load(f)
metadata["filename"] = metadata.transcriptomics_sample_id.map(transcriptomics_sample_id_file_name)
metadata.dropna(subset=["filename"], inplace=True)

metadata["T-type Label"] = metadata["T-type Label"].map(lambda x: "_".join(x.split(" ")))
my_dict = metadata.groupby("T-type Label")["filename"].apply(lambda x: x.tolist()).to_dict()

if not os.path.exists('proc/merge_bams'):
    os.makedirs('proc/merge_bams')

for key, file_list in my_dict.items():
    file_list = ["".join(["/external/rprshnas01/netdata_kcni/stlab/Nuo/STAR_for_SGSeq/coord_bams/", filename, "Aligned.sortedByCoord.out.bam"]) for filename in file_list]
    if len(file_list) > 20:
        with open("proc/merge_bams/{}.txt".format(key), "w") as f:
            for file in file_list:
                f.write(file + "\n")

if not os.path.exists('slurm_logs/merge_bams'):
    os.makedirs('slurm_logs/merge_bams')

with open("scripts/merge_bams.sh", "w") as f:
    f.write("\n".join([(\
f"""sbatch -J {key} -o slurm_logs/merge_bams/{key}.out -c 12 --mem=16000M --wrap="samtools merge -o {key}.bam -b proc/merge_bams/{key}.txt -@ 8"\
        """
        ) for key in my_dict.keys()]))