#!/usr/bin/env python3
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import numpy as np
import anndata
import src.data as sd
from scipy.sparse import csr_matrix

def main():
    # Paths to input files
    input_dir = snakemake.input.SJ_out_tabs
    gtf_path = snakemake.input.gtf_path
    path_to_manifest = snakemake.input.manifest
    path_to_metadata = snakemake.input.metadata

    # From metadata, get mapping from cell_specimen_id to ttype
    metadata = pd.read_csv(path_to_metadata)
    ID_to_ttype = metadata.loc[:, ["cell_specimen_id", "T-type Label"]]
    ID_to_ttype = ID_to_ttype[~ID_to_ttype["T-type Label"].isna()] \
        .set_index("cell_specimen_id") \
        .to_dict()["T-type Label"]

    # From manifest, get filenames of cells that have valid t-type labels (passed QC)
    flist = pd.read_csv(path_to_manifest) \
        .query("file_type == 'reverse_fastq'") \
        .query("file_name.str.contains('fastq.gz')", engine = "python") \
        .assign(file_name = lambda x : x.file_name.str.removesuffix("_R2.fastq.gz")) \
        .loc[:, ["file_name", "cell_specimen_id"]] \
        .assign(ttype = lambda x : x.cell_specimen_id.map(ID_to_ttype)) \
        .dropna(subset = ["ttype"]) \
        ["file_name"].to_list()

    print(f"{len(flist)} cells passed QC")

    # get the list of paths to SJ.out.tab files
    path_list = [Path(input_dir).joinpath(f"{directory}SJ.out.tab") for directory in flist]
    path_list = [path for path in path_list if Path(path).is_file()]
    print(f"{len(path_list)} SJ.out.tab files found out of {len(flist)} qualified cells")

    # check if the files are empty
    path_list = [path for path in path_list if Path(path).stat().st_size > 0]
    print(f"{len(path_list)} SJ.out.tab files are non-empty")

    # Use Dask to read all tab files into memory and concatenate
    combined = dd.read_csv(path_list, 
                sep = "\t",
                names = ["chromosome", "start", "end", "strand", "intron_motif", "annotation", "unique", "multi", "max_overhang"],
                dtype = {"chromosome": "object", "strand": "category", "intron_motif": "category", "annotation": "category"},
                include_path_column = True)

    # remove path prefix
    combined.path = combined.path.apply(lambda x: Path(x).name.removesuffix("SJ.out.tab"), meta=('str'))

    # removed all rows that have unique == 0
    combined = combined[combined.unique != 0]

    # filter out some uncannonical introns
    combined = combined[np.logical_not((combined.annotation == 0) & (combined.intron_motif != 0) & (combined.max_overhang < 20))]
    combined = combined[np.logical_not((combined.annotation == 0) & (combined.intron_motif == 0) & (combined.max_overhang < 30))]

    # convert dask df to pandas df and do the rest in pandas
    combined = combined.compute()

    # Obtain unique introns
    features = combined.groupby(["start", "end"], as_index = False).agg({"chromosome": "first", "start": "first", "end": "first", "strand": "first", "intron_motif": "first", "annotation": "first", "unique": "sum", "multi": "sum", "max_overhang": "first"})
    features = features.reset_index(drop = True)

    # we need to label each intron in the combined dataframe with the intron index
    features = features.assign(i = features.index)
    X = pd.merge(combined, features[["start", "end", "i"]], on = ["start", "end"], how = "inner")

    # Pivot the df to obtain intron count matrix
    X = X.pivot(index = "path", columns = "i", values = "unique")
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