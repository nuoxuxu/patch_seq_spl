import pickle
from utility.rpy2_wrapper import df_to_rdf, rdf_to_df
import pandas as pd
import numpy as np
import json
from pathlib import Path
from pybiomart import Server
from rpy2.robjects import r
from rpy2.robjects.packages import importr
for r_package in ["magrittr", "dplyr", "GenomicFeatures", "ggplot2",\
                  "stringr", "tidyr", "ggtranscript"]:
    importr(r_package)

#----------Define functions----------#
r(
    """
    gtf_path <- "/external/rprshnas01/netdata_kcni/stlab/Genomic_references/Ensembl/Mouse/Release_104/Raw/Mus_musculus.GRCm39.104.gtf"
    biotypes <- c("protein_coding", "pseudogene", "nonsense_mediated_decay", "processed_transcript")
    annotation_from_gtf <- rtracklayer::import(gtf_path) %>%
        dplyr::as_tibble() %>%
        dplyr::filter(transcript_biotype %in% biotypes) %>%
        dplyr::select(seqnames, start, end, strand, type, gene_name, transcript_name, transcript_biotype, tag)
    """
)
r("""source("/nethome/kcni/nxu/CIHR/src/helper_functions.r")""")
# Get the transcript name of the canonical transcript for each gene
server = Server(host='http://www.ensembl.org')
dataset = (server.marts['ENSEMBL_MART_ENSEMBL'].datasets['mmusculus_gene_ensembl'])
canonical = dataset.query(attributes=['external_gene_name', 'external_transcript_name'],
              filters={'transcript_is_canonical': True})
def get_canonical(gene):
    return canonical.loc[canonical["Gene name"] == gene, "Transcript name"].values[0] 
def plot_intron_group(intron_group):
    gene_name = intron_group.split("_")[0]
    r["plot_intron_group"](df_to_rdf(adata_combined.var), gene_name, get_canonical(gene_name), intron_group)

#----------Load data----------#
with open("/nethome/kcni/nxu/CIHR/proc/IC_list.json", "r") as f:
    IC_list = json.load(f)
with open("/nethome/kcni/nxu/CIHR/proc/adata_combined.pkl", "rb") as f:
    adata_combined = pickle.load(f)

#----------Plotting starts here----------#
sig_intron_attr = adata_combined.var\
    .iloc[np.argsort(adata_combined.varm["fdr"].sum(axis = 1))[::-1]]\
    .loc[:, ["intron_group", "start", "end", "gene_name"]]\
    .iloc[:np.count_nonzero(adata_combined.varm["fdr"].sum(axis=1))]
sig_intron_attr.to_csv("test.csv")

plot_intron_group("Grin1_2_25200488_-")
r["ggsave"](filename = "test.png", width = 6, height = 2)