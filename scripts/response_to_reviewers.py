from src.helper_functions import add_metadata, add_ephys_prop
import anndata
import scquint.data
import pyarrow.parquet as pq
import numpy as np
import json
import pandas as pd
import multiprocessing.pool as mp
from scquint.data import calculate_PSI

# Paths
path_to_transcriptomic_ID_ttype = "proc/transcriptomic_ID_ttype.json"
output_dir = "proc/response_to_reviewers/"
gtf_path = "/external/rprshnas01/netdata_kcni/stlab/Genomic_references/Ensembl/Mouse/Release_110/Mus_musculus.GRCm39.110.gtf"
with open("proc/IC_list.json", "r") as f:
    IC_list = json.load(f)

# read in the intron count matrix and features
X = pq.read_table(f"{output_dir}X.parquet").to_pandas()
features = pq.read_table(f"{output_dir}features.parquet").to_pandas()
features.strand.replace({"0": "NA", "1": "+", "2": "-"}, inplace=True)
transcriptomic_ID_ttype = json.load(open(path_to_transcriptomic_ID_ttype, "r"))

X = X.assign(ttype = X.index.map(transcriptomic_ID_ttype))

# randomly sample n cell(s) from each transcriptomic type
n_min_cells_per_ttype = [1, 5, 10, 20, 25, 30, 35, 40, 45, 50]
def run(params):
    n, _, group_by = params
    X_subset = X.groupby("ttype").filter(lambda x: x.shape[0] >= n)
    ind = np.hstack(X_subset.groupby("ttype").apply(lambda x: np.random.choice(x.index, size = n, replace = False)).values)
    X_subset = X_subset.loc[ind]
    X_subset = X_subset.groupby("ttype").sum()

    # construct adata object from intron count matrix
    adata = anndata.AnnData(X_subset, dtype=np.int64)
    adata.var = features

    # add gene annotation
    adata = scquint.data.add_gene_annotation(adata, gtf_path)

    # remove weird chromosomes
    adata = adata[:, np.where(adata.var.index.str.startswith("nan") == False)[0]]

    # group introns
    adata.var.strand = adata.var.strand.astype("str")
    if group_by == "five":
        adata = scquint.data.group_introns(adata, "five_prime")
    else:
        adata = scquint.data.group_introns(adata, "three_prime")

    # calculate PSI
    adata.layers["PSI"] = calculate_PSI(adata, smooth=False)
    adata = add_ephys_prop(adata)
    adata = add_metadata(adata)
    n_introns_detected = np.count_nonzero(adata.varm["fdr"].sum(axis = 0))
    return pd.Series({n: n_introns_detected})

pool = mp.Pool(processes=12)
three_results = pool.map(run, [(n, i, "three") for n in n_min_cells_per_ttype for i in range(2)])
pd.concat(three_results).to_csv(f"proc/n_introns_detected_three.csv")