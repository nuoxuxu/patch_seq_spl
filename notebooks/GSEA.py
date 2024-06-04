import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import json
import numpy as np
import sys


ephys_prop_list = pd.read_csv("/nethome/kcni/nxu/CIHR/proc/ephys_data_tt.csv", index_col=0).columns

 
# # Data Preprocessing

 
print("Read GO terms and the genes contained")


GO_MF_json = json.load(open("/nethome/kcni/nxu/CIHR/data/m5.go.mf.v2023.2.Mm.json"))
GO_dict = {}
for key in GO_MF_json.keys():
    GO_dict[key] = GO_MF_json[key]["geneSymbols"]

 
# Process observed GLM results


## concatenate individual csv files
obs_df = []
for path in [file for file in Path("/nethome/kcni/nxu/CIHR/proc/GLM_results").iterdir() if "GLM_results" in file.name]:
    temp = pd.read_csv(path, index_col=0)
    temp = temp.assign(ephys_prop = path.name.removeprefix("GLM_results_").removesuffix(".csv"))
    obs_df.append(temp)
obs_df = pd.concat(obs_df)

## Add aditional columns: gene_name, ll_ratio
obs_df = obs_df.assign(ll_ratio = 2 * (obs_df["ll"] - obs_df["ll_null"]))
obs_df = obs_df.assign(gene_name = obs_df.intron_group.str.split("_", n=1, expand=True).iloc[:, 0])
obs_ll_ratio = obs_df.groupby(["gene_name", "ephys_prop"], as_index=False).apply(lambda x: x["ll_ratio"].max())\
    .pivot(index="gene_name", columns="ephys_prop", values=None)

## Keeping intron group per gene per ephys_prop with the highest ll_ratio
obs_df["ll_ratio_rank"] = obs_df.groupby(["gene_name", "ephys_prop"])["ll_ratio"].rank(method="first", ascending=False)
obs_df = obs_df.loc[obs_df["ll_ratio_rank"] == 1]

print("Exclude genes with no GO annotations")

GO_set = set().union(*[set(GO_dict[key]) for key in GO_dict.keys()])    
unique_genes_in_adata = obs_df.gene_name
genes_in_GO_set = np.intersect1d(list(GO_set), unique_genes_in_adata)
genes_to_be_excl = np.setdiff1d(unique_genes_in_adata, list(GO_set))
obs_df = obs_df.loc[~obs_df.gene_name.isin(genes_to_be_excl)]

 
print("Preprocess GLM results with permutated ephys_prop values")


random_df = dd.read_csv("/external/rprshnas01/netdata_kcni/stlab/Nuo/random_GLM_results/*.csv", include_path_column=True)
random_df["path"] = random_df["path"].str.removeprefix("/external/rprshnas01/netdata_kcni/stlab/Nuo/random_GLM_results/").str.removesuffix(".csv")
random_df["ephys_prop"] = random_df["path"].str.rsplit("_", n=1, expand=True).iloc[:, 0]
random_df = random_df.drop("path", axis=1)
random_df = random_df.assign(ll_ratio = 2 * (random_df["ll"] - random_df["ll_null"]))
random_df = random_df.compute()

 
print("Get observed enrichment scores")


ES = []
for ephys_prop in ephys_prop_list:
    ephys_df = obs_df.loc[obs_df["ephys_prop"] == ephys_prop]
    row = {}
    for GO_term in GO_dict.keys():
        row[GO_term] = ephys_df.loc[ephys_df["gene_name"].isin(GO_dict[GO_term]), "ll_ratio"].sum()    
    ES.append(row)
ES = pd.DataFrame(ES).T
ES.columns = ephys_prop_list
ES = ES.loc[(ES.sum(axis=1) != 0)]


# for ephys_prop in ephys_prop_list:

ephys_prop = sys.argv[1]
# read all csv files of all random seeds of an ephys_prop into one df
ephys_df = dd.read_csv(f"/external/rprshnas01/netdata_kcni/stlab/Nuo/random_GLM_results/{ephys_prop}_*.csv", include_path_column=True)
ephys_df["seed"] = ephys_df["path"].str.removesuffix(".csv").str.rsplit("_", n=1, expand=True).iloc[:, 1]
ephys_df = ephys_df.drop("path", axis=1)
# filter out intron groups that are not in obs_df
ephys_df = ephys_df.loc[ephys_df["intron_group"].isin(obs_df.loc[obs_df["ephys_prop"] == ephys_prop, "intron_group"].values)]
# add ll_ratio and gene_name columns
ephys_df = ephys_df.assign(ll_ratio = 2 * (ephys_df["ll"] - ephys_df["ll_null"]))
ephys_df = ephys_df.assign(gene_name = ephys_df.intron_group.str.split("_", n=1, expand=True).iloc[:, 0])
ephys_df = ephys_df.compute()


import multiprocessing as mp

def get_ES_per_GO_term(GO_term):
    my_list = []
    for seed in range(50):
        temp = ephys_df.loc[ephys_df["seed"] == str(seed)]
        my_list.append(temp.loc[temp["gene_name"].isin(GO_dict[GO_term]), "ll_ratio"].sum())
    return my_list   

print(f"Processing {ephys_prop}...")

with mp.Pool(12) as pool:
    ES_test = pool.map_async(get_ES_per_GO_term, GO_dict.keys())
    ES_test = ES_test.get()

pd.DataFrame(dict(zip(GO_dict.keys(), ES_test))).T.to_csv(f"/nethome/kcni/nxu/CIHR/proc/ES_random/{ephys_prop}.csv")

print(f"Finished processing {ephys_prop}...")    