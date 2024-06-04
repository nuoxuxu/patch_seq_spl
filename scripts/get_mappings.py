import json
import pandas as pd

transcriptomic_id_to_specimen_id = json.load(open("proc/mappings/transcriptomic_id_to_specimen_id.json", "r"))
sample_id_to_transcriptomic_id = {key: value for value, key in transcriptomic_id_to_specimen_id.items()}
metadata = pd.read_csv("data/20200711_patchseq_metadata_mouse.csv")
transcriptomics_sample_id_file_name = metadata.set_index("transcriptomics_sample_id")["cell_specimen_id"].map(sample_id_to_transcriptomic_id).dropna().to_dict()
with open("proc/mappings/transcriptomics_sample_id_file_name.json", "w") as f:
    json.dump(transcriptomics_sample_id_file_name, f)