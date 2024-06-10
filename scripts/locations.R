library(magrittr)
library(tidyverse)
library(ggplot2)
library(reticulate)

GENOMIC_DATA_DIR <- "/Users/xunuo/Genomic_references"
gtf_path <- paste0(GENOMIC_DATA_DIR, "/Ensembl/Mouse/Release_110/Raw/Mus_musculus.GRCm39.110.gtf")
annotation_from_gtf <- rtracklayer::import(gtf_path)
annotation_from_gtf <- subset(annotation_from_gtf, mcols(annotation_from_gtf)$type %in% c("exon", "CDS"))
annotation_from_gtf <- annotation_from_gtf[mcols(annotation_from_gtf)$transcript_name %>% complete.cases(), ]

anndata <- import("anndata")
adata <- anndata$read_h5ad("proc/scquint/preprocessed_adata_three.h5ad")
sig_intron_attr <- adata$var
sig_intron_attr <- sig_intron_attr %>%
    makeGRangesFromDataFrame(keep.extra.columns = TRUE)
start(sig_intron_attr) <- start(sig_intron_attr) - 1
end(sig_intron_attr) <- end(sig_intron_attr) + 1

annotation_from_gtf[mcols(annotation_from_gtf)$transcript_name == "Kcnc1-203", ]
