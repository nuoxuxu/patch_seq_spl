import pickle
import multiprocessing as mp
import anndata
import scquint.differential_splicing as ds
from src.helper_functions import run_GLM

with open("/nethome/kcni/nxu/CIHR/proc/params.pkl", "rb") as f:
    params_list = pickle.load(f)
params_list = params_list[-200:]

def run_GLM(params):
    with open("/nethome/kcni/nxu/CIHR/proc/iterable_list.pkl", "rb") as f:
        iterable_list = pickle.load(f)
    intron_group_list = iterable_list[0]
    print(f"Starting {params[0]}_{params[1]}")
        
    path_to_adata = "/nethome/kcni/nxu/CIHR/proc/adata_filtered.h5ad"
    adata = anndata.read_h5ad(path_to_adata)

    ephys_prop = params[0]
    seed = params[1]
    adata.obs[ephys_prop] = adata.obs[ephys_prop].sample(frac=1, random_state=seed).values
    out = ds.run_regression_list(adata, ephys_prop, intron_group_list)
    out.to_csv(f"/external/rprshnas01/netdata_kcni/stlab/Nuo/random_GLM_results/{ephys_prop}_{seed}.csv", index=False)
    print(f"Done {ephys_prop}_{seed} finished")

ctx = mp.get_context("spawn")
with ctx.Pool(12) as pool:
    r = pool.map_async(run_GLM, params_list)    
    r.wait()