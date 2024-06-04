import pandas as pd
import patch_seq_spl.helper_functions as src
from importlib import reload
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt

ephys_props = pd.read_csv("data/ephys_data_sc.csv").columns[1:].to_list()

ephys_props
src.get_glm_results

def get_glm_results(path):
    import dask.dataframe as dd
    import pandas as pd
    from pathlib import Path
    from statsmodels.stats.multitest import fdrcorrection    
    glm_results = dd.read_csv([path for path in Path(path).iterdir()], include_path_column = True)\
        .rename(columns = {"Unnamed: 0": "event_name"})\
        .pivot_table(index = "event_name", columns = "path", values = "p_value").compute()
    glm_results.rename(columns = {path: Path(path).stem for path in glm_results.columns}, inplace = True)
    glm_results = glm_results.dropna()
    glm_results = pd.DataFrame(
        fdrcorrection(glm_results.values.flatten())[1].reshape(glm_results.shape), 
        index = glm_results.index, 
        columns = glm_results.columns)
    return glm_results

reload(src)

glm_results = src.get_glm_results("results/quantas")
event_name_gene_name = pd.read_csv("data/quantas/Mm.seq.all.AS.chrom.can.id2gene2symbol", sep = "\t", header = None)\
    .set_index(0)\
    .loc[:, 2]\
    .to_dict()

# Plotting
rank_by = "all"
top = 50
vmin = 0
vmax = 150

glm_results.index = glm_results.index.map(event_name_gene_name)
glm_results = glm_results.replace(0, np.nan)

VGIC_LGIC = np.load("data/VGIC_LGIC.npy", allow_pickle= True)
prop_names = json.load(open("data/mappings/prop_names.json", "r"))

p_value_matrix = src.rank_introns_by_n_sig_corr(glm_results, rank_by = rank_by, top = top)

IC_idx = np.flatnonzero(np.isin(p_value_matrix.reset_index()["event_name"].str.split("_", expand = True)[0].values, VGIC_LGIC))

# Plotting parameters
cmap = "Reds"
colorbar_label = "-log10(p-value)"
textcolors=("black", "white")
kw = dict(horizontalalignment="center", verticalalignment="center")

# Create the figure
fig, axs = plt.subplots(figsize=(10, 1+5*(top/25)), 
                        sharey=True,
                        constrained_layout=True)

# Plot the first axes (ephys_props)
im = axs.imshow(-np.log10(p_value_matrix), aspect="auto", cmap = cmap, vmin = vmin, vmax = vmax)
axs.set_xticks(np.arange(len(p_value_matrix.columns)))
axs.set_yticks(np.arange(len(p_value_matrix.index)))
axs.set_xticklabels(p_value_matrix.columns.map(prop_names), rotation=45, ha='right', fontsize = 13)

yticklabels = p_value_matrix.index.to_list()
y_labels = axs.get_yticklabels()
for i in IC_idx:
    y_labels[i].set_color("red")
axs.set_yticklabels(yticklabels)

fdr = np.vectorize({True: "*", False: " "}.get)(p_value_matrix< 0.05)
texts = []
for i in range(fdr.shape[0]):
    for j in range(fdr.shape[1]):
        kw.update(color=textcolors[int(im.norm(p_value_matrix.iloc[i, j]) > 0)])
        text = im.axes.text(j, i, fdr[i, j], **kw)
        texts.append(text) 
        
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin = vmin, vmax = vmax), 
            cmap=cmap), ax=axs, shrink=0.4, aspect = 25, location = "top", 
            pad = 0.01, label = colorbar_label)
plt.show()








with open("sbatch_scripts.txt", "w") as f:
    for prop in ephys_props:
        f.write(f'sbatch -c 12 -J {prop} -o slurm_logs/{prop}.out -t 0-5:0 --mem-per-cpu=16000 --wrap="Rscript scripts/beta_binomial.R {prop}"\n')

import anndata
adata_three = anndata.read_h5ad("proc/adata_three.h5ad")