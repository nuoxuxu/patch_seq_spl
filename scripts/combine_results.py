#!/usr/bin/env python3
import polars as pl
from pathlib import Path
import pickle
import pandas as pd

my_list = []
for path in snakemake.input:
    with open(path, "rb") as f:
        data = pickle.load(f)
        my_list.append(data)
flat_list = [item for sublist in my_list for item in sublist]
results_df = pd.concat(list(zip(*flat_list))[0])
results_df.to_csv(snakemake.output[0])
ndarray_list = list(zip(*flat_list))[1]
with open(snakemake.output[1], "wb") as f:
    pickle.dump(ndarray_list, f)