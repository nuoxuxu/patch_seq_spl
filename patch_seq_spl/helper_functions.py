import numpy as np
import pandas as pd
from typing import Literal

@pd.api.extensions.register_dataframe_accessor("glm")
class GLMAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")
    
    VGIC_LGIC = np.load("data/VGIC_LGIC.npy", allow_pickle=True)

    @staticmethod
    def get_gene_names(obj) -> list:
        return obj.index.str.split("_", expand=True).get_level_values(0).tolist()
    
    @staticmethod
    def get_VGIC_idx(obj) -> np.ndarray:
        """
        Get the indices of VGIC genes in the gene list

        Args:
            gene_list: list of gene names

        Returns:
            list of indices of VGIC genes in the gene list
        """        
        gene_names = GLMAccessor.get_gene_names(obj)
        return np.flatnonzero(pd.Series(gene_names).isin(GLMAccessor.VGIC_LGIC))
    
    def rank_introns_by_n_sig_corr(self, rank_by: str, VGIC_only: bool, sig_only: bool) -> pd.DataFrame:
        if VGIC_only:
            temp = self._obj.iloc[self.get_VGIC_idx(self._obj), :]
        else:
            temp = self._obj
        if rank_by == "all":
            p_value_matrix = temp.loc[temp.apply(lambda x: (x < 0.05)).sum(axis = 1).sort_values(ascending = False).index]
            if sig_only:
                return p_value_matrix.loc[p_value_matrix.apply(lambda x: (x < 0.05)).sum(axis=1) > 0]
            else:
                return p_value_matrix
        elif rank_by == "ephys_prop":
            ephys_props = pd.read_csv("data/ephys_data_sc.csv", index_col = 0).columns
            temp = temp[ephys_props]
            return temp.loc[temp.apply(lambda x: (x < 0.05)).sum(axis=1) > 0]
        else:
            p_value_matrix = temp.loc[temp[rank_by].abs().sort_values(ascending = True).index]
            if sig_only:
                return p_value_matrix.loc[p_value_matrix[rank_by].apply(lambda x: (x < 0.05))]
            else:
                return p_value_matrix
    
    def get_sig_gene_list(self, predictor: str, VGIC_only: bool) -> list:
        sig_glm_results = self._obj.loc[self._obj[predictor] < 0.05, predictor]
        if VGIC_only:
            gene_names = self.get_gene_names(sig_glm_results)
            return list(set(gene_names).intersection(GLMAccessor.VGIC_LGIC))
        else:
            gene_names = self.get_gene_names(sig_glm_results)
            return list(set(gene_names))

###################################################
# [1] AnnData utilities
###################################################

def add_predictors(adata):
    import json
    import warnings
    from sklearn.impute import SimpleImputer

    ephys_data_path = "data/ephys_data_sc.csv"
    transcriptomic_ID_subclass_path = "data/mappings/transcriptomic_ID_subclass.json"
    with open(transcriptomic_ID_subclass_path, "r") as f:
        transcriptomic_ID_subclass = json.load(f)
    transcriptomic_id_to_specimen_id_path = "data/mappings/transcriptomic_id_to_specimen_id.json"
    metadata_path = "data/20200711_patchseq_metadata_mouse.csv"    

    ephys_data = pd.read_csv(ephys_data_path, index_col = 0)
    ephys_data = ephys_data.loc[(np.isnan(ephys_data)).sum(axis = 1) < 6, :]
    print("Removing cells with more than 6 missing ephys properties")
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    ephys_data = pd.DataFrame(imp.fit_transform(ephys_data), index=ephys_data.index, columns=ephys_data.columns)
    print("Imputing the rest of the missing ephys values with mean")

    # keep only cells that are in both adata and ephys_data    
    adata_out = adata.copy()
    common_IDs = np.intersect1d(adata_out.obs_names, ephys_data.index)
    adata_out = adata_out[common_IDs, :]
    ephys_data = ephys_data.loc[common_IDs]

    # Add subclass labels to ephys data
    ephys_data = ephys_data.assign(subclass = ephys_data.index.map(transcriptomic_ID_subclass)).dropna()

    # Add cpms to ephys data
    cpm_path = "data/20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
    transcriptomics_sample_id_file_name_path = "data/mappings/transcriptomics_sample_id_file_name.json"
    transcriptomics_sample_id_file_name = json.load(open(transcriptomics_sample_id_file_name_path, "r"))        
    cpm = pd.read_csv(cpm_path, index_col=0)

    cpm = cpm.loc[cpm.index.isin(adata_out.var.gene_name.values.categories), :]
    cpm = cpm.T
    cpm.index = cpm.index.map(transcriptomics_sample_id_file_name)

    ephys_data = pd.concat([ephys_data, cpm.loc[ephys_data[ephys_data.index.isin(cpm.index)].index, :]], axis = 1).dropna()
    adata_out = adata_out[ephys_data.index, :]

    # Add soma depth to ephys data
    with open(transcriptomic_id_to_specimen_id_path, "r") as f:
        transcriptomic_id_to_specimen_id = json.load(f)
    metadata = pd.read_csv(metadata_path)

    specimen_id_to_transcriptomic_id = {v: k for k, v in transcriptomic_id_to_specimen_id.items()} 
    metadata = metadata.loc[~metadata["cell_soma_normalized_depth"].isna()]
    metadata["cell_specimen_id"] = metadata["cell_specimen_id"].map(specimen_id_to_transcriptomic_id)
    cell_soma_normalized_depth = metadata.dropna(subset=["cell_specimen_id"]).set_index("cell_specimen_id")["cell_soma_normalized_depth"]    

    ephys_data = ephys_data.assign(soma_depth = ephys_data.index.map(cell_soma_normalized_depth.to_dict()))

    for subclass in ephys_data["subclass"].unique():
        ephys_data[subclass] = ephys_data["subclass"] == subclass

    adata_out.obsm["predictors"] = ephys_data
    return adata_out

def update_intron_group_size(adata):
    intron_group_size = adata.var.groupby("intron_group", observed=True).count()["start"].to_frame().rename(columns={"start": "intron_group_size"})
    adata.var = pd.merge(adata.var, intron_group_size, how="left", on="intron_group")\
        .drop(columns="intron_group_size_x").rename(columns={"intron_group_size_y": "intron_group_size"})    
    return adata

def get_glm_results(path: str, key: Literal["p_value", "statistic"] = "p_value"):
    """
    Get p-values or effect sizes from likelihood ratio test
    The csv file has to contain the columns "event_name" and "p_value"

    Args:
        path: str
            path to directory containing csv files
        key: str
            "p_value" or "statistic"

    Returns:
        glm_results: pd.DataFrame
            adjusted p-values from likelihood ratio test
    """
    import dask.dataframe as dd
    import pandas as pd
    from pathlib import Path
    from statsmodels.stats.multitest import fdrcorrection

    glm_results = dd.read_csv([path for path in Path(path).iterdir()], include_path_column = True)\
        .pivot_table(index = "event_name", columns = "path", values = key).compute()
    glm_results.rename(columns = {path: Path(path).stem for path in glm_results.columns}, inplace = True)
    glm_results = glm_results.dropna()
    #TODO Why does fdrcorrection turn quantas p values all into 1?
    glm_results = pd.DataFrame(
        fdrcorrection(glm_results.values.flatten())[1].reshape(glm_results.shape), 
        index = glm_results.index, 
        columns = glm_results.columns)
    
    if "quantas" in path:
        event_name_gene_name = pd.read_csv("data/quantas/Mm.seq.all.AS.chrom.can.id2gene2symbol", sep = "\t", header = None)\
            .set_index(0)\
            .loc[:, 2]\
            .to_dict()        
        glm_results["gene_name"] = glm_results.index.map(event_name_gene_name)
        glm_results["gene_name"] = glm_results.groupby("gene_name").cumcount().add(1).astype(str).radd(glm_results["gene_name"] + '_')
        glm_results = glm_results.set_index("gene_name")

    return glm_results
        
###################################################
# [2] Plotting utilities
###################################################

def plot_glm_results(glm_results, rank_by = "all", top = 100, vmin = 0, vmax = 150, save_path = False):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import json
    import numpy as np

    prop_names = json.load(open("data/mappings/prop_names.json", "r"))
    
    glm_results = glm_results.replace(0, np.nan)
    p_value_matrix = glm_results.glm.rank_introns_by_n_sig_corr(rank_by="all", VGIC_only=False, sig_only=True)[:top]
    IC_idx = p_value_matrix.glm.get_VGIC_idx(p_value_matrix)

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

    if save_path:
        plt.savefig(save_path)

def plot_scatter_per_intron(adata, intron, ephys_prop):
    '''
    Plot scatter plot of ephys property vs PSI for a given intron

    Args:
        adata: anndata object
        intron: intron name
        ephys_prop: ephys property name 

    Returns:
        fig: a matplotlib figure object
    '''
    import matplotlib.pyplot as plt
    temp = adata.obs.index.to_series()\
        .str.split(" ", n = 1, expand = True)[0]\
        .reset_index(drop = True)
    temp = temp.groupby(temp).apply(lambda x: x.index.tolist()).to_dict()
    fig, ax  = plt.subplots()
    for subclass in temp:
        ax.plot(
            adata[temp[subclass], :].obs[ephys_prop],
            adata[temp[subclass], intron].layers["PSI"].mean(axis = 1),
            'o',
            label = subclass
        )
    fig.legend()
    ax.set_xlabel(ephys_prop)
    ax.set_ylabel(intron)
    return fig

def plot_intron_group_vs_ephys_prop(adata, intron_group, ephys_prop, grouped_by_subclass):
    import seaborn as sns
    import matplotlib.pyplot as plt

    assert adata.obsm.__contains__("predictors") == True
    
    intron_arr = adata[:, adata.var["intron_group"] == intron_group].X.toarray()
    cells_to_use = np.where(intron_arr.sum(axis=1) > 0)[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        PSI = intron_arr / intron_arr.sum(axis=1)[:, None]
    PSI = PSI[cells_to_use]
    n_classes = PSI.shape[1]

    ephys_prop_arr = adata.obsm["predictors"][[ephys_prop, "subclass"]]
    ephys_prop_arr = ephys_prop_arr.iloc[cells_to_use]
    fig, axs = plt.subplots(1, n_classes, figsize=(2+5*n_classes, 4), sharey = True)

    for i, ax in enumerate(axs):
        ax.set_xlabel("PSI")
        ax.set_ylabel(ephys_prop)
        ax.set_title(f"{intron_group} vs {ephys_prop}")
        if grouped_by_subclass == True:
            sns.scatterplot(x=PSI[:, i], y=ephys_prop_arr[ephys_prop], hue=ephys_prop_arr["subclass"], ax = ax)
        else:
            ax.scatter(PSI[:, i], ephys_prop_arr[ephys_prop].to_numpy(), marker="o", color="black", s = 5)

def save_scatter_plots_per_intron_group(adata, intron_group, ephys_prop, grouped_by_subclass):
    from pathlib import Path
    import matplotlib.pyplot as plt
    gene_name = intron_group.split("_")[0]
    plot_intron_group_vs_ephys_prop(adata, intron_group, ephys_prop, grouped_by_subclass)
    save_path = Path(f"proc/figures/{gene_name}/{intron_group}")
    if not save_path.exists():
        save_path.mkdir(parents=True)
    plt.savefig(save_path / f"{ephys_prop}.png")