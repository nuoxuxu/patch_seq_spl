import numpy as np
import pandas as pd
from typing import Literal
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

def calculate_PSI(adata, mode):
    from scquint.data import calculate_PSI
    import numpy as np
    if mode == "raw":
        return np.nan_to_num(calculate_PSI(adata, smooth=False), 0)
    elif mode == "smooth":
        return calculate_PSI(adata, smooth=True)

def get_adata_unique_intron_group(adata):
    """
    Get adata with only unique intron groups

    Args:
        adata: anndata object

    Returns:
        adata_unique_intron_group: anndata object
    """
    adata_unique_intron_group = adata[:, adata.var["i"].isin(adata.var.groupby("intron_group", observed=True).first().i.values)]
    adata_unique_intron_group.var = adata_unique_intron_group.var.reset_index()
    adata_unique_intron_group.var.index = adata_unique_intron_group.var.index.astype(str)
    return adata_unique_intron_group

def get_interactive_heatmap_(adata, corr_matrix):
    import scquint.differential_splicing as ds
    interactive_heatmap = adata.copy()
    for ephys_prop in interactive_heatmap.obsm["ephys_prop"].columns:
        interactive_heatmap.obsm[ephys_prop] = interactive_heatmap.obsm["ephys_prop"][ephys_prop].values

    del interactive_heatmap.obsm["ephys_prop"]
    interactive_heatmap = interactive_heatmap[:, interactive_heatmap.var.intron_group.isin(corr_matrix.index.to_numpy())]
    interactive_heatmap.var = interactive_heatmap.var.reset_index(drop=True)
    interactive_heatmap.var.index = interactive_heatmap.var.index.astype(str)

    for ephys_prop in list(interactive_heatmap.obsm.keys()):
        psi_list = []
        for intron_group in interactive_heatmap.var.intron_group.unique().to_numpy():
            count_arr = adata[:, adata.var.intron_group == intron_group].X.toarray()
            cells_to_use = np.flatnonzero(count_arr.sum(axis=1) != 0)
            init_psi = np.zeros((count_arr.shape[0], count_arr.shape[1]))
            df, psi = ds.run_regression(adata, intron_group, ephys_prop, subclass=False)
            init_psi[cells_to_use, :] = psi
            psi_list.append(init_psi)
        interactive_heatmap.layers[ephys_prop] = np.hstack(psi_list)
    return interactive_heatmap

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

def rank_introns_by_n_sig_corr(glm_results, rank_by):
    '''
    Rank introns by the number of significant correlations

    Args:
        glm_results: pd.DataFrame
            glm_results
        rank_by: str
            column to rank by
            "all" would rank by the sum of significant correlations across columns
        top: int or "all"
    
    Returns:
        pd.DataFrame
            p_value_matrix
    '''
    if rank_by == "all":
        p_value_matrix = glm_results.loc[glm_results.apply(lambda x: (x < 0.05)).sum(axis = 1).sort_values(ascending = False).index]
    else:
        p_value_matrix = glm_results.loc[glm_results[rank_by].abs().sort_values(ascending = True).index]
    
    return p_value_matrix

def get_sig_gene_list(glm_results, predictor):
    return list(set([x.split("_")[0] for x in glm_results.loc[glm_results[predictor] < 0.05, predictor].index.to_list()]))

def get_gene_from_intron_group(intron_group_list):
    if isinstance(intron_group_list, pd.core.indexes.base.Index):
        return intron_group_list.str.split("_", expand=True).get_level_values(0).tolist()
    else:
        return [x.split("_")[0] for x in intron_group_list]

def get_VGIC_idx(gene_list):
    """
    Get the indices of VGIC genes in the gene list

    Args:
        gene_list: list of gene names

    Returns:
        list of indices of VGIC genes in the gene list
    """
    VGIC_LGIC = np.load("data/VGIC_LGIC.npy", allow_pickle=True)
    if isinstance(gene_list, list) or isinstance(gene_list, np.ndarray):
        gene_list = pd.Series(gene_list) 
    return np.flatnonzero(gene_list.isin(VGIC_LGIC))
        
###################################################
# [2] Plotting utilities
###################################################

def plot_glm_results(glm_results, rank_by = "all", top = 100, vmin = 0, vmax = 150, save_path = False):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import json
    import numpy

    VGIC_LGIC = np.load("data/VGIC_LGIC.npy", allow_pickle= True)
    prop_names = json.load(open("data/mappings/prop_names.json", "r"))
    
    glm_results = glm_results.replace(0, np.nan)
    p_value_matrix = rank_introns_by_n_sig_corr(glm_results, rank_by = rank_by)[:top]

    try:
        IC_idx = np.flatnonzero(np.isin(p_value_matrix.reset_index()["event_name"].str.split("_", expand = True)[0].values, VGIC_LGIC))
    except KeyError:
        IC_idx = np.flatnonzero(np.isin(p_value_matrix.reset_index()["gene_name"].str.split("_", expand = True)[0].values, VGIC_LGIC))

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

def plot_intron_group_vs_ephys_prop(adata, intron_group, ephys_prop):
    import matplotlib.pyplot as plt
    intron_arr = adata[:, adata.var["intron_group"] == intron_group].X.toarray()
    cells_to_use = np.where(intron_arr.sum(axis=1) > 0)[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        PSI = intron_arr / intron_arr.sum(axis=1)[:, None]
    PSI = PSI[cells_to_use]
    ephys_prop_arr = adata.obsm[ephys_prop]
    ephys_prop_arr = ephys_prop_arr[cells_to_use]

    fig, axs = plt.subplots(1, 5, figsize=(2+5*5, 5), sharey = True)

    for i, ax in enumerate(axs):
        ax.set_xlabel("PSI")
        ax.set_ylabel(ephys_prop)
        ax.set_title(f"{intron_group} vs {ephys_prop}")
        ax.scatter(PSI[:, i], ephys_prop_arr, marker="o", color="black", s = 5)
    # return ax.scatter(PSI, ephys_prop_arr, marker="o", color="black", s = 5)  

def add_gene_annotation(features, gtf_path, filter_unique_gene=True):
    import warnings
    if not gtf_path.endswith("110.gtf"):
        warnings.warn("The file path does not end with '110.gtf'", UserWarning)
    gtf = pd.read_csv(
        gtf_path,
        sep="\t",
        header=None,
        comment="#",
        names=[
            "chromosome",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attribute",
        ],
        dtype={'chromosome': str}
    )

    gtf = gtf[gtf.feature == "exon"]
    gtf["gene_id"] = gtf.attribute.str.extract(r'gene_id "([^;]*)";')
    gtf["gene_name"] = gtf.attribute.str.extract(r'gene_name "([^;]*)";')
    gtf["canonical"] = gtf["attribute"].str.findall("Ensembl_canonical").apply(lambda x: len(x) > 0)

    if np.array([x.startswith("chr") for x in features.chromosome.unique()]).sum() != 0:
        gtf.chromosome = "chr" + gtf.chromosome.astype(str)
    else:
        pass
    gene_id_name = gtf[["gene_id", "gene_name"]].drop_duplicates()

    exon_starts = (
        gtf[["chromosome", "start", "gene_id", "canonical"]].copy().rename(columns={"start": "pos"})
    )
    exon_starts.pos -= 1
    exon_ends = (
        gtf[["chromosome", "end", "gene_id", "canonical"]].copy().rename(columns={"end": "pos"})
    )
    exon_ends.pos += 1
    exon_boundaries = pd.concat(
        [exon_starts, exon_ends], ignore_index=True
    )
    genes_by_exon_boundary = exon_boundaries.groupby(["chromosome", "pos"])\
        .agg({"canonical": "any", "gene_id": lambda x: x.unique()})

    features = (
        features.merge(
            genes_by_exon_boundary,
            how="left",
            left_on=["chromosome", "start"],
            right_on=["chromosome", "pos"],
        )
        .rename(columns={"gene_id": "gene_id_start", "canonical": "canonical_start"})
        .set_index(features.index)
    )
    features = (
        features.merge(
            genes_by_exon_boundary,
            how="left",
            left_on=["chromosome", "end"],
            right_on=["chromosome", "pos"],
        )
        .rename(columns={"gene_id": "gene_id_end", "canonical": "canonical_end"})
        .set_index(features.index)
    )

    def fill_na_with_empty_array(val):
        return val if isinstance(val, np.ndarray) else np.array([])
    def fill_na_with_empty_string(val):
        return val if not np.isnan(val) else ""    
    features.gene_id_start = features.gene_id_start.apply(fill_na_with_empty_array)
    features.gene_id_end = features.gene_id_end.apply(fill_na_with_empty_array)
    features.canonical_start = features.canonical_start.apply(fill_na_with_empty_string)
    features.canonical_end = features.canonical_end.apply(fill_na_with_empty_string)
    features["gene_id_list"] = features.apply(
        lambda row: np.unique(np.concatenate([row.gene_id_start, row.gene_id_end])),
        axis=1,
    )
    features["n_genes"] = features.gene_id_list.apply(len)
    features.gene_id_list = features.gene_id_list.apply(
        lambda x: ",".join(x.tolist())
    )
    features.gene_id_start = features.gene_id_start.apply(
        lambda x: ",".join(x.tolist())
    )
    features.gene_id_end = features.gene_id_end.apply(
        lambda x: ",".join(x.tolist())
    )

    if filter_unique_gene:
        print("Filtering to introns associated to 1 and only 1 gene.")
        features = features[features.n_genes == 1]
        features["gene_id"] = features.gene_id_list
        features.drop(columns=["gene_id_list",], inplace=True)
        features = features.merge(gene_id_name, how="left", on="gene_id").set_index(
            features.index
        )
        features.index = features.gene_name.astype(str) + "_" + features.index.astype(str)
    return features

###################################################
# [3] Benchmarking inclusion criteria for introns
###################################################

def count_n_valid_PSI(intron):
    return np.sum(np.logical_and(intron != 0, intron != 1))

def get_sum_stats(intron_matrix, features, n_cells_per_ttype, params):
    from scquint.data import group_normalize
    n_min_SJ = params[0]
    n_min_cell = params[1]
    min_obs = params[2]
    # filter by n_min_SJ, remove introns that have less than n_min_SJ SJs in total
    intron_matrix = intron_matrix.loc[:, ~(intron_matrix.sum(axis=0) < n_min_SJ)]
    features = features.loc[features["i"].isin(intron_matrix.columns.astype(int))]

    # filter by n_min_cell cells, remove ttypes that have less than n_min_cell cells
    ttypes_to_keep = n_cells_per_ttype.index[np.where(n_cells_per_ttype >= n_min_cell)[0]]
    intron_matrix = intron_matrix.loc[ttypes_to_keep]

    # remove introns not expressed in this subset of cells
    # X_filtered = X_filtered[(X_filtered.sum(axis=1) != 0)]

    # calculate PSI
    PSI = group_normalize(intron_matrix.values, features.intron_group.values)
    PSI = np.nan_to_num(PSI)
    
    # count number of valid PSI
    n_valid_PSI = (np.apply_along_axis(count_n_valid_PSI, 0, PSI) >= min_obs).sum()

    # calculate some summary statistics
    n_total_cell = n_cells_per_ttype.loc[ttypes_to_keep].sum().values[0]
    n_ttype = len(ttypes_to_keep)

    return {"n_min_SJ": n_min_SJ, "n_min_cell": n_min_cell, "min_obs": min_obs,
        "n_ttype": n_ttype, "n_total_cell": n_total_cell, "n_valid_PSI": n_valid_PSI}

def get_sum_stats_sc(adata, params):
    import scquint.data as sd
    adata = adata.copy()
    min_global_SJ_counts, min_cells_per_feature = params
    
    # filter out introns at min global SJ counts
    adata = sd.filter_min_global_SJ_counts(adata, min_global_SJ_counts)
    n_intron_after_min_global_SJ = adata.shape[1]

    # filter out introns at min cells per feature
    adata = sd.filter_min_cells_per_feature(adata, min_cells_per_feature)
    n_intron_final = adata.shape[1]

    return {"min_global_SJ_counts": min_global_SJ_counts,
            "min_cells_per_feature": min_cells_per_feature,
            "n_introns": adata.shape[1], "n_intron_after_min_global_SJ": n_intron_after_min_global_SJ,
            "n_intron_final": n_intron_final}

def filter_adata(adata, params):
    import scquint.data as sd
    min_global_SJ_counts, min_cells_per_feature, min_cells_per_intron_group = params
    # filter out introns at min global SJ counts
    adata = adata.copy()
    adata = sd.filter_min_global_SJ_counts(adata, min_global_SJ_counts)
    
    # filter out introns at min cells per feature and intron group
    adata = sd.filter_min_cells_per_feature(adata, min_cells_per_feature)
    adata = sd.filter_min_cells_per_intron_group(adata, min_cells_per_intron_group)
    # adata.var = adata.var.reset_index(drop = True)
    return adata    

def get_XYW(ttype_by_intron, features, n_cells_per_ttype, params):
    from scquint.data import group_normalize, make_intron_group_summation_cpu
    import numpy_groupies as npg
    n_min_SJ = params[0]
    n_min_cell = params[1]
    min_obs = params[2]

    # filter by n_min_SJ, remove introns that have less than n_min_SJ SJs in total
    ttype_by_intron = ttype_by_intron.loc[:, ~(ttype_by_intron.sum(axis=0) < n_min_SJ)]
    features = features.loc[features["i"].isin(ttype_by_intron.columns.astype(int))]

    # filter by n_min_cell cells, remove ttypes that have less than n_min_cell cells
    ttypes_to_keep = n_cells_per_ttype.index[np.where(n_cells_per_ttype >= n_min_cell)[0]]
    ttype_by_intron = ttype_by_intron.loc[ttypes_to_keep]

    # calculate PSI
    PSI = group_normalize(ttype_by_intron.values, features.intron_group.values)
    
    # assign 0 to NaN values
    PSI = np.nan_to_num(PSI)

    # remove introns with less than min_obs
    index_to_keep = np.apply_along_axis(count_n_valid_PSI, 0, PSI) >= min_obs
    PSI = PSI[:, index_to_keep]
    features = features.loc[index_to_keep]
    
    # Get Y
    ephys_tt = pd.read_csv("proc/ephys_data_tt.csv", index_col = 0)    
    Y = ephys_tt.loc[ttypes_to_keep].values 

    # Get W
    groups = pd.factorize(features.intron_group.values)[0]
    intron_group_sums = PSI @ make_intron_group_summation_cpu(groups)    
    W = np.vstack([intron_group_sums[i][[groups]] for i in range(intron_group_sums.shape[0])])
    return PSI, Y, W

def get_XYW_random(cell_by_intron, features, ttypes, params):
    from scquint.data import group_normalize, make_intron_group_summation_cpu
    n_min_SJ = params[0]
    n_min_cell = params[1]
    min_obs = params[2]
    n_cell_to_sample = params[3]
    assert n_cell_to_sample <= n_min_cell

    # filter by n_min_SJ
    introns_to_keep = np.flatnonzero(cell_by_intron.sum(axis=0) >= n_min_SJ)
    cell_by_intron = cell_by_intron[:, introns_to_keep]
    features = features.iloc[introns_to_keep]

    # remove cells that are in ttypes with fewer than 50 cells
    cells_to_keep = ttypes.groupby("ttype").filter(lambda x: len(x) >= n_min_cell).index
    ttypes = ttypes.loc[cells_to_keep].reset_index(drop = True)
    cell_by_intron = cell_by_intron[cells_to_keep]

    # randomly select 5 cells from each ttype
    selected_cells = ttypes.groupby("ttype").apply(lambda x: x.sample(5)).index.get_level_values(1)
    ttypes = ttypes.iloc[selected_cells]["ttype"]
    cell_by_intron = cell_by_intron[selected_cells]

    ttype_by_intron = npg.aggregate(pd.factorize(ttypes)[0], cell_by_intron.toarray(), axis = 0)

    PSI = group_normalize(ttype_by_intron, features.intron_group.values)

    # assign 0 to NaN values
    PSI = np.nan_to_num(PSI)

    # remove introns with less than min_obs
    index_to_keep = np.apply_along_axis(count_n_valid_PSI, 0, PSI) >= min_obs
    PSI = PSI[:, index_to_keep]
    ttype_by_intron = ttype_by_intron[:, index_to_keep]
    features = features.loc[index_to_keep]

    # Get Y
    ephys_tt = pd.read_csv("proc/ephys_data_tt.csv", index_col = 0)    
    Y = ephys_tt.loc[np.unique(ttypes)].values 

    # Get W
    groups = pd.factorize(features.intron_group.values)[0]
    intron_group_sums = ttype_by_intron @ make_intron_group_summation_cpu(groups)    
    W = np.vstack([intron_group_sums[i][[groups]] for i in range(intron_group_sums.shape[0])])
    return PSI, Y, W

def permutation_test(x, y, n_permutations=1000):
    n, m = len(x), len(y)
    t_obs = np.abs(np.mean(x) - np.mean(y))
    combined = np.concatenate([x, y])
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        x_perm = combined[:n]
        y_perm = combined[n:]
        t_perm = np.abs(np.mean(x_perm) - np.mean(y_perm))
        if t_perm > t_obs:
            count += 1
    return count / n_permutations

def weighted_corr(X, Y, W, rank=False):
    X = X.T
    W = W.T
    if rank == True:
        from scipy.stats import rankdata
        X = rankdata(X, axis = 1, nan_policy='omit')
        Y = rankdata(Y, axis = 0, nan_policy='omit')
    weighted_corr_row = []
    for i in range(X.shape[0]):   
        X_subset = X[i, :]
        X_subset = np.nan_to_num(X_subset)
        Y_subset = Y
        W_subset = W[i, :]  
        with np.errstate(divide='ignore'):
            X_mean = (X_subset @ W_subset) / W_subset.sum()
            Y_mean = (Y_subset.T @ W_subset) / W_subset.sum()
        X_subset = X_subset - X_mean
        Y_subset = Y_subset - Y_mean
        cov_XY = ((X_subset*W_subset)@Y_subset)/np.sum(W_subset)
        cov_XX = ((X_subset*W_subset)@X_subset)/np.sum(W_subset)
        cov_YY = np.diagonal(Y_subset.T@(Y_subset*W_subset[:, None])/np.sum(W_subset))
        with np.errstate(divide='ignore'):
            weighted_corr = cov_XY / np.sqrt(cov_XX * cov_YY)
        weighted_corr_row.append(weighted_corr)
    c = np.vstack(weighted_corr_row)
    return c    

def run_GLM(params):
    import pickle
    import anndata
    import scquint.differential_splicing as ds
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

def get_beta_coefficients(adata, ephys_prop, intron_group):
    return ds.run_regression_2(adata.X, adata.obs[ephys_prop], adata.var, intron_group)    

def print_process_id(i):
    import multiprocessing as mp
    return {i: mp.current_process().pid}