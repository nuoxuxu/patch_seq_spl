#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
from scquint.data import load_adata_from_starsolo, add_gene_annotation, group_introns, calculate_PSI
import matplotlib.pyplot as plt
import anndata as ad

# Paths to input files
gtf_path = "/external/rprshnas01/netdata_kcni/stlab/Genomic_references/Ensembl/Human/Release_103/Raw/Homo_sapiens.GRCh38.103.gtf"
path_to_metadata = "/external/rprshnas01/netdata_kcni/stlab/Nuo/CIHR/Berg_STARsolo_outputs/20200625_patchseq_metadata_human.csv"
path_to_manifest = "/external/rprshnas01/external_data/aibs/AIBS_patchseq_2020/human/2021-03-04_human_file_manifest.csv"

adata_human = load_adata_from_starsolo("/external/rprshnas01/netdata_kcni/stlab/Nuo/CIHR/Berg_STARsolo_outputs/output/SJ/raw")
metadata = pd.read_csv(path_to_metadata)

manifest = pd.read_csv(path_to_manifest)
manifest = manifest.loc[~manifest.file_name.str.endswith("Aligned.sortedByCoord.out.bam")]
manifest = manifest.loc[~manifest.file_name.str.endswith("_R2.fastq.gz")]
manifest["file_name"] = manifest["file_name"].str.removesuffix("_R1.fastq.gz")
cell_specimen_id_to_file_name = manifest.set_index("cell_specimen_id")["file_name"].to_dict()

# add cell type labels
metadata = metadata.assign(file_name = metadata.cell_specimen_id.map(cell_specimen_id_to_file_name))
file_name_to_ttype = metadata.set_index("file_name")["corresponding_AIT2.3.1_alias"].to_dict()
adata_human.obs = adata_human.obs.assign(ttype = adata_human.obs.index.map(file_name_to_ttype))


# generate barplot that shows the number of cells per cell type
series_to_plot = adata_human.obs.groupby("ttype").size()
series_to_plot.index = series_to_plot.index.str.split(" ", expand=True).get_level_values(3)
fig, ax = plt.subplots(figsize = (6, 5), layout="tight")
sns.barplot(series_to_plot, x = series_to_plot.index, y = series_to_plot.values, ax=ax)
ax.set_xticklabels(series_to_plot.index)
ax.set_ylabel("Number of cells per cell type")
ax.set_title("Human Patch-seq glutamatergic cell type distribution")

# plot PSI distribution at the single-cell level
adata_sc = adata_human.copy()
adata_sc.var.chromosome = adata_sc.var.chromosome.astype("str")
adata_sc = add_gene_annotation(adata_sc, gtf_path)

adata3_sc = group_introns(adata_sc, "three_prime")
adata5_sc = group_introns(adata_sc, "five_prime")

adata3_sc.layers["PSI"] = calculate_PSI(adata3_sc, smooth=False)
adata5_sc.layers["PSI"] = calculate_PSI(adata5_sc, smooth=False)

adata3_sc.layers["PSI"] = np.nan_to_num(adata3_sc.layers["PSI"])
adata5_sc.layers["PSI"] = np.nan_to_num(adata5_sc.layers["PSI"])

# plot PSI distribution at the cell type level
adata_tt = pd.DataFrame(adata_human.X.toarray(), index=adata_human.obs.index, columns=adata_human.var.index)
adata_tt = adata_tt.assign(ttype = adata_tt.index.map(file_name_to_ttype)).groupby("ttype").sum()
adata_tt = ad.AnnData(adata_tt, obs = pd.DataFrame(index=adata_tt.index), var = adata_human.var)

adata_tt.var.chromosome = adata_tt.var.chromosome.astype("str")
adata_tt = add_gene_annotation(adata_tt, gtf_path)

adata3_tt = group_introns(adata_tt, "three_prime")
adata5_tt = group_introns(adata_tt, "five_prime")

adata3_tt.layers["PSI"] = calculate_PSI(adata3_tt, smooth=False)
adata5_tt.layers["PSI"] = calculate_PSI(adata5_tt, smooth=False)

adata3_tt.layers["PSI"] = np.nan_to_num(adata3_tt.layers["PSI"])
adata5_tt.layers["PSI"] = np.nan_to_num(adata5_tt.layers["PSI"])

# plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (5, 6), layout="tight", sharex=True)
ax1.hist(adata3_sc.layers["PSI"].flatten(), bins=100)
ax1.set_title("Human Patch-seq PSI distribution at the single-cell level")
ax1.set_ylabel("Number of SJ")
ax2.hist(adata3_tt.layers["PSI"].flatten(), bins=100)
ax2.set_title("Human Patch-seq PSI distribution at the cell type level")
ax2.set_ylabel("Number of SJ")
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.set_xlabel("PSI")
plt.show()


