library(ggtranscript)
library(magrittr)
library(tidyverse)
library(ggplot2)
library(rtracklayer)
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

findAdjacent <- function(query, subject) {
    hits <- findOverlaps(query, subject)
    get_width <- function(x) {
        query[x[1]]
        n <- length(subject[[x[2]]])
        width(pintersect(rep(query[x[1]], n), subject[[x[2]]], drop.nohit.ranges=TRUE)) %>% unique()
    }
    ranges_to_keep <- apply(as.data.frame(hits), 1, get_width) %>% lapply(function(x) 1 %in% x) %>% unlist() %>% which()
    hits[ranges_to_keep, ]
}

get_base_plot <- function(annodation_gene_name) {
    exons <- subset(annodation_gene_name, mcols(annodation_gene_name)$type == "exon") %>%
        as.data.frame()
    cds <- subset(annodation_gene_name, mcols(annodation_gene_name)$type == "CDS") %>%
        as.data.frame()
    exons %>%
        ggplot(
            aes(
                xstart = start,
                xend = end,
                y = transcript_name
            ),
            position_jitter()
        ) +
        geom_range(
            height = 0.25
        ) +
        geom_range(
            data = cds,
            aes(fill = tag)
        ) +
        geom_intron(
            data = to_intron(exons, "transcript_name"),
            aes(strand = strand),
            arrow.min.intron.length = 500
        )
}

plot_intron_group <- function(my_intron_group, adjacent_only = TRUE) {
    my_gene_name <- str_split(my_intron_group, "_")[[1]][1]

    sig_intron_attr_subset <- sig_intron_attr %>%
        subset(mcols(.)$intron_group == my_intron_group)

    annotation_for_gene <- annotation_from_gtf %>%
        subset(mcols(.)$gene_name == my_gene_name)

    exonByTranscript <- split(annotation_for_gene, mcols(annotation_for_gene)$transcript_name)

    if (adjacent_only) {
        hits <- findAdjacent(sig_intron_attr_subset, exonByTranscript)
    } else {
        hits <- findOverlaps(sig_intron_attr_subset, exonByTranscript)
    }

    transcripts_to_plot <- names(exonByTranscript)[unique(subjectHits(hits))]

    junctions <- as.data.frame(sig_intron_attr_subset)[queryHits(hits), ]
    junctions$transcript_name <- names(exonByTranscript)[subjectHits(hits)]

    exonByTranscript[transcripts_to_plot] %>%
        unlist(use.names = FALSE) %>%
        get_base_plot() +
        geom_junction(data = junctions, junction.y.max = 0.5)
}