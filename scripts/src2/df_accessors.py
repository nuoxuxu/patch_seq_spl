import numpy as np
import pandas as pd
from typing import Literal
import anndata
import json

metadata = pd.read_csv("data/20200711_patchseq_metadata_mouse.csv")

with open("data/mappings/transcriptomic_ID_subclass.json", "r") as f:
    transcriptomic_id_to_subclass = json.load(f)

with open("data/mappings/transcriptomic_id_to_specimen_id.json", "r") as f:
    transcriptomic_id_to_specimen_id = json.load(f)

with open("data/mappings/transcriptomics_sample_id_file_name.json", "r") as f:
    sample_id_file_name = json.load(f)

specimen_id_to_transcriptomic_id = {str(int(key)): value for value, key in transcriptomic_id_to_specimen_id.items()}

transcriptomic_id_cell_type = metadata\
    .assign(file_name = lambda x: x["transcriptomics_sample_id"].map(sample_id_file_name))\
    .set_index("file_name")\
    ["T-type Label"].to_dict()

file_id_to_cell_type = pd.read_csv("data/20200711_patchseq_metadata_mouse.csv")\
    .set_index("ephys_session_id")\
    ["T-type Label"].to_dict()

file_name_to_subclass = pd.read_csv("data/2021-09-13_mouse_file_manifest.csv", dtype={"cell_specimen_id": str})\
    .query("technique == 'intracellular_electrophysiology'")\
    .assign(transcriptomic_id = lambda x:x["cell_specimen_id"].map(specimen_id_to_transcriptomic_id))\
    .assign(subclass=lambda x: x["transcriptomic_id"].map(transcriptomic_id_to_subclass))\
    .dropna(subset=["subclass"])\
    .set_index("file_name")\
    ["subclass"].to_dict()

file_name_to_cell_type = pd.read_csv("data/2021-09-13_mouse_file_manifest.csv", dtype={"cell_specimen_id": str})\
    .query("technique == 'intracellular_electrophysiology'")\
    .assign(cell_type = lambda x: x["file_id"].map(file_id_to_cell_type))\
    .assign(cell_type = lambda x: x["cell_type"].str.replace(" ", "_"))\
    .set_index("file_name")\
    ["cell_type"].to_dict()

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
                return p_value_matrix.loc[p_value_matrix.apply(lambda x: (x < 0.05)).sum(axis = 1) > 0]
            else:
                return p_value_matrix
        elif rank_by == "ephys_prop":
            ephys_props = pd.read_csv("data/ephys_data_sc.csv", index_col = 0).columns
            p_value_matrix = temp.loc[temp[ephys_props].apply(lambda x: (x < 0.05)).sum(axis = 1).sort_values(ascending = False).index]
            if sig_only:
                return p_value_matrix.loc[p_value_matrix[ephys_props].apply(lambda x: (x < 0.05)).sum(axis = 1) > 0]
            else:
                return p_value_matrix
        elif rank_by == "subclass":
            subclass_list = ["Vip", "Sst", "Pvalb", "Lamp5", "Sncg", "Serpinf1"]
            p_value_matrix = temp.loc[temp[subclass_list].apply(lambda x: (x < 0.05)).sum(axis = 1).sort_values(ascending = False).index]
            if sig_only:
                return p_value_matrix.loc[p_value_matrix[subclass_list].apply(lambda x: (x < 0.05)).sum(axis = 1) > 0]
            else:
                return p_value_matrix
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
            return list(set(gene_names).intersection(self.VGIC_LGIC))
        else:
            gene_names = self.get_gene_names(sig_glm_results)
            return list(set(gene_names))

    def plot_logp(self, intron_group):
        import seaborn as sns
        import matplotlib.pyplot as plt

        temp = self._obj.loc[intron_group].apply(lambda x: -np.log10(x))

        fig, ax = plt.subplots()
        temp.sort_values(ascending=False)\
            .pipe(sns.barplot, ax=ax)
        ax.set_ylabel("-log10(p-value)")
        ax.set_xlabel("predictors")
        ax.set_title(f"{intron_group}")
        g = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    def plot_heatmap(self, rank_by = "all", top = 100, vmin = 0, vmax = 150, save_path = False):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import json
        import numpy as np

        prop_names = json.load(open("data/mappings/prop_names.json", "r"))
        
        p_value_matrix = self._obj.replace(0, np.nan)
        p_value_matrix = p_value_matrix[:top]
        IC_idx = self.get_VGIC_idx(p_value_matrix)

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

@pd.api.extensions.register_dataframe_accessor("ipfx")
class IPFXAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @staticmethod
    def colnames_to_subclass(obj):
        out = obj.copy()
        out.columns = out.columns.map(file_name_to_subclass)
        out = out.loc[:, out.columns.notnull()]
        return out
    
    @staticmethod
    def colnames_to_cell_types(obj):
        out = obj.copy()
        out.columns = out.columns.map(file_name_to_cell_type)
        out = out.loc[:, out.columns.notnull()]
        return out

    def plot_ephys_traces(self, cell_type_1: str, cell_type_2: str, window: int = 500):
        """
        Plot overlayed ephys traces of two cell types

        Args:
            cell_type_1: str
                cell type 1
            cell_type_2: str
                cell type 2
            window: int
                window size

        Returns:
            fig: matplotlib.pyplot.figure
                figure object
            ax: matplotlib.pyplot.axis
                axis object
        """
        import matplotlib.pyplot as plt

        arr_1 = self.colnames_to_cell_types(self._obj)[cell_type_1].values[:window, :]
        arr_2 = self.colnames_to_cell_types(self._obj)[cell_type_2].values[:window, :]

        fig, ax = plt.subplots()
        for i in range(arr_1.shape[1]):
            ax.plot(arr_1[:, i], color = "blue", alpha = 0.1)
        for i in range(arr_2.shape[1]):
            ax.plot(arr_2[:, i], color = "red", alpha = 0.1)
        leg = ax.legend([cell_type_1, cell_type_2]) 
        leg.legend_handles[0].set_color('blue')
        leg.legend_handles[1].set_color('red')
        ax.xaxis.set_label_text("Time (ms)")
        ax.yaxis.set_label_text("Voltage (mV)")    
    