from ipfx.dataset.create import create_ephys_data_set
from ipfx.utilities import drop_failed_sweeps
from ipfx.data_set_features import extract_data_set_features
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ipfx.feature_extractor import SpikeFeatureExtractor, SpikeTrainFeatureExtractor
from ipfx.stimulus_protocol_analysis import LongSquareAnalysis
from ipfx.epochs import get_stim_epoch
import numpy as np
import sys
from collections import defaultdict

def get_rheobase_number(dataset):
    drop_failed_sweeps(dataset)
    (cell_features, sweep_features, cell_record, sweep_records, cell_state, feature_states) = extract_data_set_features(dataset)
    rheo_sweep_number = cell_features["long_squares"]["rheobase_sweep"]["sweep_number"]
    return rheo_sweep_number

def get_rheobase_threshold_idx(dataset):
    long_square_table = dataset.filtered_sweep_table(stimuli=dataset.ontology.long_square_names)
    long_square_sweeps = dataset.sweep_set(long_square_table.sweep_number)
    rheo_sweep_number = get_rheobase_number(dataset)

    stim_start_index, stim_end_index = get_stim_epoch(long_square_sweeps.i[0])
    stim_start_time = long_square_sweeps.t[0][stim_start_index]
    stim_end_time = long_square_sweeps.t[0][stim_end_index]
    spfx = SpikeFeatureExtractor(start=stim_start_time, end=stim_end_time)
    sptfx = SpikeTrainFeatureExtractor(start=stim_start_time, end=stim_end_time)
    long_square_analysis = LongSquareAnalysis(spfx, sptfx, subthresh_min_amp=-100.0)
    out = long_square_analysis.analyze(long_square_sweeps)
    return out["spikes_set"][long_square_table[long_square_table.sweep_number == rheo_sweep_number].index[0]].threshold_index.iloc[0]

def extract_rheobase_sweep(dataset, window=1000, align_to_threshold_idx=True):
    rheo_sweep_number = get_rheobase_number(dataset)
    rheo_sweep = dataset.sweep(rheo_sweep_number)
    if align_to_threshold_idx:
        threshold_idx = get_rheobase_threshold_idx(dataset)
        return rheo_sweep.t[threshold_idx:threshold_idx + window], rheo_sweep.v[threshold_idx:threshold_idx + window]
    return rheo_sweep.t, rheo_sweep.v

def main():
    input_dir = sys.argv[1]
    output_path = sys.argv[2]

    path_list = list(Path(input_dir).iterdir())

    my_dict = defaultdict(list)
    for i, path in enumerate(path_list):
        dataset = create_ephys_data_set(str(path))
        drop_failed_sweeps(dataset)
        t, v = extract_rheobase_sweep(dataset)
        my_dict[path.name] = np.array(v)

    np.savez(output_path, **my_dict)

if __name__ == "__main__":
    main()