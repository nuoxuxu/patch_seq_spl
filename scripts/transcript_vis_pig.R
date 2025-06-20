# Load packages
suppressMessages(
    {
        library(ggplot2)
        library(rtracklayer)
        library(stringr)
        library(glue)
        library(readr)
        library(dplyr)
        library(ggtranscript)
        library(GenomicRanges)
    }
)

# Load data
gtf_path <- "lampert_data/Sus_scrofa.Sscrofa11.1.105.chr.gtf"
annotation_from_gtf <- rtracklayer::import(gtf_path)
annotation_from_gtf <- subset(annotation_from_gtf, mcols(annotation_from_gtf)$type %in% c("exon", "CDS"))
annotation_from_gtf <- annotation_from_gtf[mcols(annotation_from_gtf)$transcript_name %>% complete.cases(), ]

SJ_attribue_path <- "lampert_data/SJ_attributes.csv"
SJ_attributes  <- read_csv("lampert_data/SJ_attributes.csv")

ephys_prop <- read_csv("lampert_data/ephys_prop.csv")

corr_p_value_matrix <- read_csv("lampert_data/corr_p_value_matrix.csv")

ratio_matrix <- read_csv("lampert_data/ratio_matrix.csv")

# Define functions
get_base_plot <- function(annodation_gene_name) {
    library(ggtranscript)
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
            data = cds
        ) +
        geom_intron(
            data = to_intron(exons, "transcript_name"),
            aes(strand = strand),
            arrow.min.intron.length = 500
        )
}

# Start here
SJ <- SJ_attributes %>% 
    filter(SJ == "SNAP25_17:19214960_+_19211274")

get_base_plot(subset(annotation_from_gtf, mcols(annotation_from_gtf)$gene_name=="SNAP25")) +
    geom_junction(
        data = mutate(SJ, transcript_name = "SNAP25-204"),
        aes(
            xstart = start,
            xend = end,
            strand = strand
        ),
        junction.y.max = 0.5,
        fill = "red"
    )
