import numpy as np
import pandas as pd
import anndata
import json

metadata = pd.read_csv("data/20200711_patchseq_metadata_mouse.csv")

with open("data/mappings/transcriptomics_sample_id_file_name.json", "r") as f:
    sample_id_file_name = json.load(f)

transcriptomic_id_cell_type = metadata\
    .assign(file_name = lambda x: x["transcriptomics_sample_id"].map(sample_id_file_name))\
    .set_index("file_name")\
    ["T-type Label"].to_dict()

class ExtendedAnnData(anndata.AnnData):
    def __init__(self, adata):
        super().__init__(adata.X, obs=adata.obs, var=adata.var, obsm=adata.obsm, varm=adata.varm, uns=adata.uns)

    def filter_adata(self, params):
        import src.data as sd
        min_global_SJ_counts, min_cells_per_feature, min_cells_per_intron_group = params

        self = sd.filter_min_global_SJ_counts(self, min_global_SJ_counts)
        self = sd.filter_min_cells_per_feature(self, min_cells_per_feature)
        self = sd.filter_min_cells_per_intron_group(self, min_cells_per_intron_group)
        return ExtendedAnnData(self)

    def update_intron_group_size(self):
        intron_group_size = self.var.groupby("intron_group", observed=True).count()["start"].to_frame().rename(columns={"start": "intron_group_size"})
        self.var = pd.merge(self.var, intron_group_size, how="left", on="intron_group")\
            .drop(columns="intron_group_size_x").rename(columns={"intron_group_size_y": "intron_group_size"})    
        return self    
    
    def add_predictors(self):
        import json
        import warnings
        from sklearn.impute import SimpleImputer

        ephys_data_path = "data/ephys_data_sc.csv"
        transcriptomic_ID_subclass_path = "data/mappings/transcriptomic_ID_subclass.json"
        with open(transcriptomic_ID_subclass_path, "r") as f:
            transcriptomic_ID_subclass = json.load(f)
        with open("data/mappings/transcriptomics_file_name_cell_type.json") as f:
            transcriptomics_file_name_cell_type = json.load(f)
        transcriptomic_id_to_specimen_id_path = "data/mappings/transcriptomic_id_to_specimen_id.json"
        metadata_path = "data/20200711_patchseq_metadata_mouse.csv"    

        ephys_data = pd.read_csv(ephys_data_path, index_col = 0)
        ephys_data = ephys_data.loc[(np.isnan(ephys_data)).sum(axis = 1) < 6, :]
        print("Removing cells with more than 6 missing ephys properties")
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        ephys_data = pd.DataFrame(imp.fit_transform(ephys_data), index=ephys_data.index, columns=ephys_data.columns)
        print("Imputing the rest of the missing ephys values with mean")

        # keep only cells that are in both adata and ephys_data    
        common_IDs = np.intersect1d(self.obs_names, ephys_data.index)
        self = self[common_IDs, :]
        ephys_data = ephys_data.loc[common_IDs]

        # Add subclass and cell type labels to ephys data
        ephys_data = ephys_data.assign(subclass = ephys_data.index.map(transcriptomic_ID_subclass)).dropna()
        ephys_data = ephys_data.assign(cell_type = ephys_data.index.map(transcriptomics_file_name_cell_type)).dropna()

        # Add cpms to ephys data
        cpm_path = "data/20200513_Mouse_PatchSeq_Release_cpm.v2.csv"
        transcriptomics_sample_id_file_name_path = "data/mappings/transcriptomics_sample_id_file_name.json"
        transcriptomics_sample_id_file_name = json.load(open(transcriptomics_sample_id_file_name_path, "r"))        
        cpm = pd.read_csv(cpm_path, index_col=0)

        cpm = cpm.loc[cpm.index.isin(self.var.gene_name.values.categories), :]
        cpm = cpm.T
        cpm.index = cpm.index.map(transcriptomics_sample_id_file_name)

        ephys_data = pd.concat([ephys_data, cpm.loc[ephys_data[ephys_data.index.isin(cpm.index)].index, :]], axis = 1).dropna()
        self = self[ephys_data.index, :]

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
        for cell_type in ephys_data["cell_type"].unique():
            ephys_data[cell_type] = ephys_data["cell_type"] == cell_type

        self.obsm["predictors"] = ephys_data
        return ExtendedAnnData(self)

    def get_ttype_by_prop(self):
        out = self.obsm["predictors"]\
            .assign(t_type = lambda x: x.index.map(transcriptomic_id_cell_type))\
            .dropna(subset=["t_type"])\
            .drop(columns = ['Sst', 'Pvalb', 'Vip', 'Lamp5', 'Sncg', 'Serpinf1', 'subclass'])\
            .groupby("t_type")\
            .median()
        return out

    def get_ttype_by_SJ(self):
        import numpy_groupies as npg
        ttype_by_SJ = npg.aggregate(
            pd.factorize(self.obs.index.map(transcriptomic_id_cell_type))[0], 
            self.X.toarray(),axis = 0).astype(int)

        return pd.DataFrame(
            ttype_by_SJ,
            index = pd.factorize(self.obs.index.map(transcriptomic_id_cell_type))[1],
            columns = self.var_names)

    def plot_SJ_prop_sc(self, intron_group, ephys_prop, grouped_by_subclass):
        import seaborn as sns
        import matplotlib.pyplot as plt

        assert self.obsm.__contains__("predictors") == True
        
        intron_arr = self[:, self.var["intron_group"] == intron_group].X.toarray()
        cells_to_use = np.where(intron_arr.sum(axis=1) > 0)[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            PSI = intron_arr / intron_arr.sum(axis=1)[:, None]
        PSI = PSI[cells_to_use]
        n_classes = PSI.shape[1]

        ephys_prop_arr = self.obsm["predictors"][[ephys_prop, "subclass"]]
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

    def save_scatter_plots_per_intron_group(self, intron_group, ephys_prop, grouped_by_subclass):
        from pathlib import Path
        import matplotlib.pyplot as plt
        
        gene_name = intron_group.split("_")[0]
        self.plot_SJ_prop_sc(intron_group, ephys_prop, grouped_by_subclass)
        save_path = Path(f"proc/figures/{gene_name}/{intron_group}")
        if not save_path.exists():
            save_path.mkdir(parents=True)
        plt.savefig(save_path / f"{ephys_prop}.png")    
    
    def plot_SJ_prop_sc_ttype(self, intron_group, ephys_prop):
        import plotly.express as px
        import matplotlib.pyplot as plt

        assert self.obsm.__contains__("predictors") == True

        ttype_by_prop = self.get_ttype_by_prop()
        ttype_by_SJ = self.get_ttype_by_SJ()
        
        SJ_idx = self[:, self.var.intron_group == intron_group].var.index.astype(int)
        ttype_by_SJ = ttype_by_SJ.iloc[:, SJ_idx]

        with np.errstate(divide='ignore'):
            ttype_by_SJ_arr = ttype_by_SJ.values / ttype_by_SJ.sum(axis=1).values[:, None]

        n_classes = ttype_by_SJ_arr.shape[1]

        df_for_plotting = pd.DataFrame(ttype_by_SJ_arr, index = ttype_by_SJ.index)\
            .assign(ephys_prop = ttype_by_prop[ephys_prop].values)\
            .reset_index()\
            .melt(id_vars=["ephys_prop", "index"], var_name="SJ_idx")

        return px.scatter(df_for_plotting,
                          x="value", y="ephys_prop", facet_col="SJ_idx", 
                          hover_name = df_for_plotting["index"], 
                          labels = {"value": "PSI", "ephys_prop": ephys_prop})
        
    def plot_ggtranscript(self, intron_group, adjacent_only = True, focus = True, transcripts_subset = [0], fill_by = "tag"):
        from src.ryp import r, to_r, to_py
        to_r(transcripts_subset, "transcripts_subset")
        r("transcripts_subset <- unlist(transcripts_subset)")
        to_r(adjacent_only, "adjacent_only")
        to_r(focus, "focus")

        if to_py('!exists("sig_intron_attr")'):                
            r("source('scripts/transcript_viz.r')")
            r("annotation_from_gtf <- get_annotation_from_gtf()")
            r("sig_intron_attr <- get_sig_intron_attr()")
            r(f"plot_intron_group('{intron_group}', adjacent_only=adjacent_only, focus=focus, transcripts_subset=transcripts_subset, fill_by='{fill_by}')")
        else:
            r(f"plot_intron_group('{intron_group}', adjacent_only=adjacent_only, focus=focus, transcripts_subset=transcripts_subset, fill_by='{fill_by}')")