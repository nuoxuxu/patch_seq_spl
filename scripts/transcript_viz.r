library(ggtranscript)
library(magrittr)
library(dplyr)
library(ggplot2)
library(rtracklayer)
library(reticulate)

# intron_group_list <- read.csv("proc/to_R/corr_matrix.csv", row.names = "intron_group") %>% rownames()
# gene_list <- sapply(strsplit(intron_group_list, "_"), "[", 1)

gtf_path <- "/scratch/s/shreejoy/nxu/Genomic_references/mm39/Mus_musculus.GRCm39.110.gtf"
biotypes <- c("protein_coding", "pseudogene", "nonsense_mediated_decay", "processed_transcript")
annotation_from_gtf <- rtracklayer::import(gtf_path) %>%
    dplyr::as_tibble() %>%
    dplyr::filter(transcript_biotype %in% biotypes) %>%
    dplyr::filter(gene_name %in% gene_list) %>%
    dplyr::filter(tag == "Ensembl_canonical") %>% 
    dplyr::select(seqnames, start, end, strand, type, gene_name, transcript_name, transcript_biotype, tag)

get_base_plot <- function(annodation_gene_name) {
    exons <- annodation_gene_name %>% filter(type == "exon")
    # obtain cds
    cds <- annodation_gene_name %>% filter(type == "CDS")
    # Differentiating UTRs from the coding sequence
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
            fill = "white",
            height = 0.25
        ) +
        geom_range(
            data = cds,
        ) +
        geom_intron(
            data = to_intron(exons, "transcript_name"),
            aes(strand = strand),
            arrow.min.intron.length = 500
        )
}

plot_intron_group <- function(sig_intron_attr, transcript_name, intron_group, xmin = NULL, xmax = NULL) {
    gene_of_interest <- strsplit(intron_group, "_")[[1]][[1]]
    annotation <- annotation_from_gtf %>%
        filter(gene_name == gene_of_interest, transcript_name == {{transcript_name}})
    sig_intron_attr_subset <- sig_intron_attr %>%
        filter(intron_group == {{intron_group}})
    if(is.null(xmin)) {
        xmin <- annotation %>%
            filter(end == min(unique(pull(sig_intron_attr_subset, start)))-1) %>%
            pull(start) %>%
            min()
    }
    if (is.infinite(xmin)) {
        xmin <- min(unique(pull(sig_intron_attr_subset, start)))
    }
    if(is.null(xmax)) {
        xmax <- annotation %>%
            filter(start == max(unique(pull(sig_intron_attr_subset, end)))+1) %>%
            pull(end) %>%
            max()
    }
    junctions <- sig_intron_attr_subset %>%
        filter(intron_group == {{intron_group}}) %>%
        mutate(transcript_name = {{transcript_name}})
    g <- annotation %>%
        get_base_plot() +
            geom_junction(data = junctions, junction.y.max = 0.5) +
            scale_color_manual(values = c("True" = "red", "False" = "blue")) +
            # geom_junction(data = junctions, junction.y.max = 0.5, color = "#006eff") +
            coord_cartesian(xlim = c(xmin, xmax)) +
            labs(title = intron_group) +
            theme_light()+
            theme(legend.position = "none") +
            geom_text(
                data = add_exon_number(filter(annotation, type == "exon"), "transcript_name"),
                aes(x = (start + end) / 2, label = exon_number),
                size = 3.5,
                nudge_y = 0.4
                )
    g <- g + ylab(NULL) + xlab(NULL)
    return(g)
}

g <- plot_intron_group(sig_intron_attr, "Kcnt1-201", "Kcnt1_2_25778080_+")

ggsave("Kcnt1_2_25778080_+.png", g, width = 10, height = 5, units = "in", dpi = 300)