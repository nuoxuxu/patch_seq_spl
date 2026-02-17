#!/usr/bin/env python3
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import numpy as np
import anndata
import src.data as sd
from scipy.sparse import csr_matrix
import os

def main():
    sample_sheet = snakemake.input['sample_sheet']
    gtf_path = snakemake.config['gtf_path']

    sample_sheet = sample_sheet[~sample_sheet["cell_type_label"].isna()]

    path_list = sample_sheet["path"].to_list()

    path_list = [path for path in path_list if Path(path).stat().st_size > 0]
    print(f"{len(path_list)} SJ.out.tab files are non-empty")

    # Use Dask to read all tab files into memory and concatenate
    combined = dd.read_csv(path_list, 
                sep = "\t",
                names = ["chromosome", "start", "end", "strand", "intron_motif", "annotation", "unique", "multi", "max_overhang"],
                dtype = {"chromosome": "object", "strand": "category", "intron_motif": "category", "annotation": "category"},
                include_path_column = True)

    # removed all rows that have unique == 0
    combined = combined[combined.unique != 0]

    # filter out some uncannonical introns
    combined = combined[np.logical_not((combined.annotation == 0) & (combined.intron_motif != 0) & (combined.max_overhang < 20))]
    combined = combined[np.logical_not((combined.annotation == 0) & (combined.intron_motif == 0) & (combined.max_overhang < 30))]

    # convert dask df to pandas df and do the rest in pandas
    combined = combined.compute()

    # Add cell_specimen_id and cell_type_label columns
    combined = pd.merge(combined, sample_sheet[["path", "cell_specimen_id", "cell_type_label"]], on = "path")

    # Obtain unique introns
    features = combined.groupby(["start", "end"], as_index = False).agg({"chromosome": "first", "start": "first", "end": "first", "strand": "first", "intron_motif": "first", "annotation": "first", "unique": "sum", "multi": "sum", "max_overhang": "first"})
    features = features.reset_index(drop = True)

    # we need to label each intron in the combined dataframe with the intron index
    features = features.assign(i = features.index)
    X = pd.merge(combined, features[["start", "end", "i"]], on = ["start", "end"], how = "inner")

    # Pivot the df to obtain intron count matrix
    X = X.pivot(index = "cell_specimen_id", columns = "i", values = "unique")
    X.fillna(0, inplace = True)

    # construct adata object from X and features
    adata = anndata.AnnData(csr_matrix(X))
    adata.obs_names = X.index.to_list()
    adata.var = features

    # add gene annotation
    adata = sd.add_gene_annotation(adata, gtf_path, filter_unique_gene=True)

    # change strand annotation
    adata.var.strand = adata.var.strand.replace({"1": "+", "2": "-"})

    # group introns by sharing three prime splice site
    adata = sd.group_introns(adata, snakemake.wildcards.group_by)

    # remove introns from weird chromosomes
    adata = adata[:, ~(pd.isnull(adata.var.gene_name) == True)]

    # h5ad cannot have columns with mixied types
    adata.var.canonical_end = adata.var.canonical_end.map({True: 1, False: 0, "": 2})
    adata.var.canonical_start = adata.var.canonical_start.map({True: 1, False: 0, "": 2})

    # Write adata objects to disk
    adata.write(snakemake.output[0])

if __name__ == "__main__":
    main()