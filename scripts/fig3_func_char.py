import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import anndata

#------------functions----------------
def check_region(row):
    cluster_start = row.start_x
    if np.isnan(row.start_y):
        return "Non-coding"
    if row.strand_y == "+":
        if cluster_start < row.start_y:
            return "5' UTR"
        if cluster_start < row.end_y:
            return "CDS"
        return "3' UTR"
    elif row.strand_y == "-":
        if cluster_start < row.start_y:
            return "3' UTR"
        if cluster_start < row.end_y:
            return "CDS"
        return "5' UTR"
    else:
        raise Exception("strand not implemented")  
        
def get_one_intron_per_event_adata(adata, std_threshold = False):
    if std_threshold:
        adata = adata[:, adata.layers["PSI"].std(axis = 0) > std_threshold]
    introns = adata.var \
        .assign(std = adata.X.std(axis = 0))\
        .reset_index(names = "intron")\
        .groupby("intron_group", group_keys=False, as_index = False).apply(lambda x : x.iloc[np.argmax(x["std"])]).intron.values
    cleaned_adata = adata[:, introns]
    return cleaned_adata  

def get_dict(adata, std_threshold = False):
    adata.var["sig"] = adata.varm["fdr"].sum(axis = 1) > 0
    one_intron_per_event_adata = get_one_intron_per_event_adata(adata, std_threshold = std_threshold)
    temp = adata.var \
        .groupby("intron_group", group_keys=False, as_index = False)\
        .agg({"start":"min", "end": "max"})\
        .merge(one_intron_per_event_adata.var[["intron_group", "gene_id", "strand", "sig"]], on = "intron_group") \
        .merge(gene_cds, on = "gene_id", how = "left")
    my_dict = {}
    my_dict["all"] = temp.apply(check_region, axis = 1).value_counts().values
    my_dict["subset"] = temp.loc[temp.sig == True].apply(check_region, axis = 1).value_counts().values
    return my_dict
#---------annotation----------------
summary = pd.read_excel("/nethome/kcni/nxu/CIHR/data/Non_redundant_cass_mm10_summary.xlsx")

df = pd.read_csv(
    "/external/rprshnas01/netdata_kcni/stlab/Genomic_references/Ensembl/Mouse/Release_104/Raw/Mus_musculus.GRCm39.104.gtf",
    '\t', header=None, comment="#",
    names=['chromosome', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'],
    )
df = df[df.feature=="CDS"]
df['gene_id'] = df.attribute.str.extract(r'gene_id "([^;]*)";')
gene_cds = df.groupby("gene_id").agg({"chromosome": "first", "start": "min", "end": "max", "strand": "first"})
#---------Input data----------------
with open("/external/rprshnas01/netdata_kcni/stlab/Nuo/CIHR/proc/adata_three_sc.pkl", "rb") as f:
    adata3_sc = pickle.load(f)
with open("/external/rprshnas01/netdata_kcni/stlab/Nuo/CIHR/proc/adata_five_sc.pkl", "rb") as f:
    adata5_sc = pickle.load(f)    
with open("/external/rprshnas01/netdata_kcni/stlab/Nuo/CIHR/proc/adata_three_tt.pkl", "rb") as f:
    adata3_tt = pickle.load(f)
with open("/external/rprshnas01/netdata_kcni/stlab/Nuo/CIHR/proc/adata_five_tt.pkl", "rb") as f:
    adata5_tt = pickle.load(f)        
imputed_Y_sc = anndata.read_h5ad("/external/rprshnas01/netdata_kcni/stlab/Nuo/CIHR/proc/imputedY.h5ad")  
imputed_Y_ttype = anndata.read_h5ad("/external/rprshnas01/netdata_kcni/stlab/Nuo/CIHR/proc/imputedY_ttype.h5ad")  

# some preprocessing
adata3_sc_dict = get_dict(adata3_sc, std_threshold=0.1)
adata5_sc_dict = get_dict(adata5_sc, std_threshold=0.1)
adata3_tt_dict = get_dict(adata3_tt, std_threshold=0.1)
adata5_tt_dict = get_dict(adata5_tt, std_threshold=0.1)    
adata_labels = ["CDS", "5' UTR", "3' UTR", "Non-coding"]

# imputed_Y_sc
temp = pd.merge(imputed_Y_sc.var, summary[["name", "NMD classification"]], left_index = True, right_on = "name", how = "inner").set_index("name")
imputed_Y_sc = imputed_Y_sc[:, temp.index]
imputed_Y_sc.var = temp

imputed_Y_sc_dict = {}
imputed_Y_sc_dict["all"] = imputed_Y_sc.var["NMD classification"].value_counts().values
imputed_Y_sc_dict["subset"] = imputed_Y_sc[:, np.flatnonzero(imputed_Y_sc.varm["fdr"].sum(axis = 1))].var["NMD classification"].value_counts().values
imputed_Y_sc_labels = imputed_Y_sc.var["NMD classification"].unique().tolist()

# imputed_Y_ttype
temp = pd.merge(imputed_Y_ttype.var, summary[["name", "NMD classification"]], left_index = True, right_on = "name", how = "inner").set_index("name")
imputed_Y_ttype = imputed_Y_ttype[:, temp.index]
imputed_Y_ttype.var = temp

imputed_Y_ttype_dict = {}
imputed_Y_ttype_dict["all"] = imputed_Y_ttype.var["NMD classification"].value_counts().values
imputed_Y_ttype_dict["subset"] = imputed_Y_ttype[:, np.flatnonzero(imputed_Y_ttype.varm["fdr"].sum(axis = 1))].var["NMD classification"].value_counts().values
imputed_Y_ttype_labels = imputed_Y_ttype.var["NMD classification"].unique().tolist()

#------------plot----------------
adata3_tt_dict["subset"] = np.array([272,  44,   6, 0])
fig, axs = plt.subplots(6, 1, figsize = (3, 15), layout = "constrained", sharey = True)
for lables, values, ax in zip([adata_labels, adata_labels, adata_labels, adata_labels, imputed_Y_sc_labels, imputed_Y_ttype_labels], 
                              [adata3_sc_dict, adata3_tt_dict, adata5_sc_dict, adata5_tt_dict, imputed_Y_sc_dict, imputed_Y_ttype_dict], 
                                axs.flat):
        if type(values) == np.ndarray:
            ax.bar(lables, values)
        if type(values) == dict:
            bottom = np.zeros(4)
            for boolean, weight_count in values.items():
                ax.bar(lables, weight_count, 0.5, label=boolean, bottom=bottom)
                bottom += weight_count
        ax.set_xticklabels(lables, rotation = 45, ha = "right")
        ax.set_ylabel("Number of introns")
        ax.set_title("All introns")        
plt.savefig("/nethome/kcni/nxu/CIHR/results/figures/func_char_2.png")