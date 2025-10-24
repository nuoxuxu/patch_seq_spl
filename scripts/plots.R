library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggtranscript)
library(GenomicRanges)
library(stringr)
library(ggpubr)

# Load data

corr_p_value_matrix <- read_csv("lampert_data/corr_p_value_matrix.csv")
ratio_matrix <- read_csv("lampert_data/ratio_matrix.csv")
ephys_props <- read_csv("lampert_data/ephys_prop.csv")
ratio_matrix_pb <- read_csv("lampert_data/ratio_matrix_pb.csv")

annotation_from_gtf <- rtracklayer::import("lampert_data/Sus_scrofa.Sscrofa11.1.105.chr.gtf")
annotation_from_gtf <- subset(annotation_from_gtf, mcols(annotation_from_gtf)$type %in% c("exon", "CDS"))
annotation_from_gtf <- annotation_from_gtf[mcols(annotation_from_gtf)$transcript_name %>% complete.cases(), ]

SJ_attributes  <- read_csv("lampert_data/SJ_attributes.csv") %>%
    mutate(
        SJ_group = str_remove(SJ, "_[^_]*$"),
        .keep = "all"
    )

# Define functions

ratio_matrix <- ratio_matrix * 10

plot_scatter <- function(ephys, SJ_of_interest) {
    df <- tibble(
        ratio = pull(ratio_matrix, {{SJ_of_interest}}),
        ephys_props = pull(ephys_props, {{ephys}})
    ) 
    # df %>%
    #     ggplot(aes(x = ratio, y = ephys_props)) +
    #     geom_point() +
    #     xlab("Ratio") +
    #     ylab(ephys) +
    #     ggtitle(SJ_of_interest) +
    #     theme_minimal()
    df %>% ggscatter(
        x = "ratio",
        y = "ephys_props",
        add = "reg.line",
        conf.int = TRUE,
        cor.coef = TRUE,
        cor.method = "pearson",
        xlab = "Ratio",
        ylab = ephys,
        title = SJ_of_interest
    )
    ggsave(glue::glue("results/plots/{SJ_of_interest}_{ephys}.png"), width = 8, height = 6)
}

plot_SJ <- function(SJ_of_interest) {
    ephys_props_of_interest <- corr_p_value_matrix %>%
        filter(SJ == SJ_of_interest) %>%
        select(-c(SJ, sum)) %>%
        pivot_longer(cols = everything(), names_to = "gene", values_to = "p_value") %>%
        filter(p_value == TRUE) %>%
        pull(gene)
    lapply(ephys_props_of_interest, plot_scatter, SJ_of_interest)
}

plot_scatter("PC1", "SCN9A_15:72912040_-_72846418")

get_base_plot <- function(annodation_gene_name) {
    exons <- annodation_gene_name %>% filter(type == "exon")
    cds <- annodation_gene_name %>% filter(type == "CDS")
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
SJ_of_interest <- "SCN9A_15:72912040_-_72846418"
cell_type_1 <- "C-OSMR-SST"
cell_type_2 <- "C-TAC1-LRP1B"
ephys_prop <- "PC1"
transcript_name <- "SCN9A-201"


annodation_gene_name <- subset(annotation_from_gtf, mcols(annotation_from_gtf)$transcript_name=="SCN9A-201") %>%
    as.data.frame() %>%
    filter(type %in% c("exon", "CDS"))

annodation_gene_name <- annodation_gene_name %>%
    mutate(
        cell_type = !!cell_type_1
    )

SJ_group_of_interest <- SJ_attributes %>%
    filter(SJ == !!SJ_of_interest) %>%
    pull(SJ_group)

# transpose
pb_ratio <- ratio_matrix_pb %>%
    pivot_longer(cols = -labels, names_to = "SJ", values_to = "ratio") %>%
    filter(labels %in% !!cell_type_1) %>% 
    select(-labels)

# Cell type 1

SJ_df <- SJ_attributes %>%
    filter(SJ_group == !!SJ_group_of_interest) %>%
    mutate(transcript_name = !!transcript_name) %>%
    mutate(cell_type = !!cell_type_1) %>% 
    left_join(
        pb_ratio,
        by = c("SJ" = "SJ")
    )

SJ_df %>% write_csv("SJ_df.csv")
SJ_df <- read_csv("SJ_df.csv")

# remove y-axis title
get_base_plot(annodation_gene_name) +
    geom_junction(
        data = SJ_df,
        aes(size = ratio),
        junction.y.max = 0.5
    ) +
    ylab(cell_type_1) +
    theme_minimal()

ggsave(glue::glue("results/plots/ggtranscript/{SJ_of_interest}_{cell_type_1}.png"), width = 8, height = 2.5)

# Cell type 2
SJ_df <- SJ_attributes %>%
    filter(SJ_group == !!SJ_group_of_interest) %>%
    mutate(transcript_name = !!transcript_name) %>%
    mutate(cell_type = !!cell_type_2) %>% 
    left_join(
        pb_ratio,
        by = c("SJ" = "SJ")
    )

SJ_df %>% write_csv("SJ_df.csv")
SJ_df <- read_csv("SJ_df.csv")

annodation_gene_name <- subset(annotation_from_gtf, mcols(annotation_from_gtf)$gene_name=="SCN9A") %>%
    as.data.frame() %>%
    filter(type %in% c("exon", "CDS"))

get_base_plot(annodation_gene_name) +
    geom_junction(
        data = SJ_df,
        aes(size = ratio),
        junction.y.max = 0.5
    ) +
    ylab(cell_type_1) +
    theme_minimal()
ggsave("test.png")
SJ_list <- read_csv("lampert_data/SJ_list.csv", col_names=FALSE) %>% pull(X1)

sapply(SJ_list, plot_SJ)
plot_scatter("APD", "SCN9A_15:72912040_-_72846418")
