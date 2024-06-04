import pandas as pd
import scquint.differential_splicing as sd
import json
import pickle
import sys

# Import adata
with open("proc/adata_filtered.pkl", "rb") as f:
    adata = pickle.load(f)

df_list = []
for intron_group in adata.var["intron_group"].unique().to_numpy():
    df_intron_group, psi = sd.run_regression(adata, intron_group, sys.argv[1], subclass=True)
    df_list.append(df_intron_group)

df = pd.concat(df_list)

df.sort_values("p_value").to_csv(f"/scratch/s/shreejoy/nxu/CIHR/proc/GLM_subclass/{sys.argv[1]}.csv")