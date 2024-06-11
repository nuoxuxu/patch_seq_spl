suppressMessages(
    {
        library(ggtranscript)
        library(magrittr)
        library(tidyverse)
        library(ggplot2)
        library(rtracklayer)
        library(reticulate)
        library(stringr)
        library(glue)
    }
)

get_annotation_from_gtf <- function(gtf_path) {
    annotation_from_gtf <- rtracklayer::import(gtf_path)
    annotation_from_gtf <- subset(annotation_from_gtf, mcols(annotation_from_gtf)$type %in% c("exon", "CDS"))
    annotation_from_gtf <- annotation_from_gtf[mcols(annotation_from_gtf)$transcript_name %>% complete.cases(), ]
    annotation_from_gtf
}

get_sig_intron_attr <- function(sig_intron_attr) {
    sig_intron_attr <- sig_intron_attr %>%
        makeGRangesFromDataFrame(keep.extra.columns = TRUE)
    start(sig_intron_attr) <- start(sig_intron_attr) - 1
    end(sig_intron_attr) <- end(sig_intron_attr) + 1
    sig_intron_attr
}

findAdjacent <- function(query, subject) {
    if (class(query) == "GRanges" && class(subject) == "CompressedGRangesList") {
        hits <- findOverlaps(query, subject)
        get_width <- function(x) {
            query[x[1]]
            n <- length(subject[[x[2]]])
            width(pintersect(rep(query[x[1]], n), subject[[x[2]]], drop.nohit.ranges=TRUE)) %>% unique()
        }
        ranges_to_keep <- apply(as.data.frame(hits), 1, get_width) %>% lapply(function(x) 1 %in% x) %>% unlist() %>% which()
        hits[ranges_to_keep, ]
    } else if (class(query) == "GRanges" && class(subject) == "GRanges") {
        hits <- findOverlaps(query, subject)
        ranges_to_keep <- pintersect(query[queryHits(hits)], subject[subjectHits(hits)]) %>%
            width() %>%
            lapply(function(x) 1 %in% x) %>%
            unlist() %>%
            which()
        hits[ranges_to_keep, ]
    }
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

plot_intron_group <- function(my_intron_group, adjacent_only = TRUE, focus = TRUE) {
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

    # Get junctions for plotting junctions
    junctions <- as.data.frame(sig_intron_attr_subset)[queryHits(hits), ]
    junctions$transcript_name <- names(exonByTranscript)[subjectHits(hits)]

    transcripts_to_plot <- names(exonByTranscript)[unique(subjectHits(hits))]
    exonByTranscript <- exonByTranscript[transcripts_to_plot]

    # Get xmin and xmax for the longest intron
    longest_intron <- sig_intron_attr_subset[which.max(width(sig_intron_attr_subset)), ]
    downstream_exon_idx <- precede(longest_intron, unlist(exonByTranscript), ignore.strand = TRUE)
    upstream_exon_idx <- follow(longest_intron, unlist(exonByTranscript), ignore.strand = TRUE)

    # The upstream exon is the exon preceding the donor exon if the splice site is annotated
    # If upstream_exon_idx is not defined, i.e., ASE or novel TSS
    if (is.na(upstream_exon_idx)) {
        xmin <- start(longest_intron) - width(longest_intron)
        # If upstream_exon_idx is defined, i.e., get the upstream exon
    } else if (!is.na(upstream_exon_idx)) {
        upstream_exon <- unlist(exonByTranscript)[upstream_exon_idx]
        # If the exon preceding the upstream exon is not defined, i.e., the upstream exon is the first unannotated exon
        if (!is.na(follow(upstream_exon, unlist(exonByTranscript), ignore.strand = TRUE))) {
            xmin <- unlist(exonByTranscript)[follow(upstream_exon, unlist(exonByTranscript), ignore.strand = TRUE)] %>% start()
        # if the exon preceding the upstream exon is defined
        } else if (is.na(follow(upstream_exon, unlist(exonByTranscript), ignore.strand = TRUE))) {
            xmin <- start(upstream_exon)
        }
    }

    # The downstream exon is the exon following the acceptor exon if the splice site is annotated
    # If downstream_exon_idx is not defined, i.e., AEE or novel TTS
    if (is.na(downstream_exon_idx)) {
        xmax <- end(longest_intron) + width(longest_intron)
        # If downstream_exon_idx is defined, i.e., get the downstream exon
    } else if (!is.na(downstream_exon_idx)) {
        downstream_exon <- unlist(exonByTranscript)[downstream_exon_idx]
        # If the exon following the downstream exon is not defined, i.e., the downstream exon is the last unannotated exon
        if (!is.na(precede(downstream_exon, unlist(exonByTranscript), ignore.strand = TRUE))) {
            xmax <- unlist(exonByTranscript)[precede(downstream_exon, unlist(exonByTranscript), ignore.strand = TRUE)] %>% end()
        } else if (is.na(precede(downstream_exon, unlist(exonByTranscript), ignore.strand = TRUE))) {
            xmax <- end(downstream_exon)
        }
    }
    if (focus) {
        exonByTranscript %>%
            unlist(use.names = FALSE) %>%
            get_base_plot() +
            geom_junction(data = junctions, junction.y.max = 0.5) +
            coord_cartesian(xlim = c(xmin, xmax))
    } else {
        exonByTranscript %>%
            unlist(use.names = FALSE) %>%
            get_base_plot() +
            geom_junction(data = junctions, junction.y.max = 0.5)
    }
}

plot_per_gene <- function(x, adjacent_only = TRUE, focus = TRUE) {
    gene_name <- str_split(unname(x[1]), "_")[[1]][1]
    for (intron in x) {
        path <- glue("proc/figures/{gene_name}/{intron}")
        if (!dir.exists(path)) {
            dir.create(path, recursive = TRUE)
        }
        plot_intron_group(intron, adjacent_only = adjacent_only, focus = focus)
        ggsave(glue("proc/figures/{gene_name}/{intron}/transcript_viz.png"))
    }
}

# sig_VGIC_SJ <- read.csv("sig_VGIC_SJ.csv")
# intron_group_list <- sig_VGIC_SJ$event_name
# gene_name_list <- sapply(str_split(intron_group_list, "_"), `[`, 1)
# intron_by_gene_name <- split(intron_group_list, gene_name_list)
# lapply(intron_by_gene_name, plot_per_gene)

GENOMIC_DATA_DIR <- "/Users/xunuo/Genomic_references"
gtf_path <- paste0(GENOMIC_DATA_DIR, "/Ensembl/Mouse/Release_110/Raw/Mus_musculus.GRCm39.110.gtf")

sig_intron_attr <- get_sig_intron_attr(sig_intron_attr)
annotation_from_gtf <- get_annotation_from_gtf(gtf_path)