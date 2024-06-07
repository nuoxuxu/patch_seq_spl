library(ggtranscript)
library(magrittr)
library(tidyverse)
library(ggplot2)
library(rtracklayer)
library(reticulate)

# Define functions
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

get_transcripts_to_plot <- function(intron_group) {
    get_unique_combinations <- function(input_list) {
        unique_combinations <- list()
        for (name in names(input_list)) {
            combination <- input_list[[name]]
            if (all(combination == FALSE)) next
            combination_str <- paste(combination, collapse = "_")
            if (!combination_str %in% names(unique_combinations)) {
            unique_combinations[[combination_str]] <- c(name)
            } else {
            unique_combinations[[combination_str]] <- c(unique_combinations[[combination_str]], name)
            }
        }
        return(unique_combinations)
    }
    annotation <- annotation_from_gtf %>%
        filter(gene_name == strsplit(intron_group, "_")[[1]][1])
    start_sites <- sig_intron_attr %>% 
        filter(intron_group == {{intron_group}}) %>% 
        pull(start)    
    unique_combinations <- lapply(split(annotation$end, as.factor(annotation$transcript_name)), function(x) (start_sites - 1) %in% x) %>% 
        get_unique_combinations()
    lapply(unique_combinations, function(x) {x[[1]]}) %>%
        unlist() %>%
        unname()
}

plot_intron_group <- function(intron_group, xmin = NULL, xmax = NULL) {
    annotation <- annotation_from_gtf %>%
        filter(transcript_name %in% get_transcripts_to_plot({{intron_group}}))
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
    if (is.null(xmax)) {
        xmax <- annotation %>%
            filter(start == max(unique(pull(sig_intron_attr_subset, end)))+1) %>%
            pull(end) %>%
            max()
    }
    junctions <- annotation %>% 
        mutate(end = end + 1) %>%
        filter(type == "exon") %>%
        select(c(end, transcript_name)) %>% 
        left_join(sig_intron_attr_subset, join_by(end == start)) %>% 
        drop_na() %>% 
        dplyr::rename(start = `end.y`) %>% 
        dplyr::select(c(start, end, transcript_name, strand))
    annotation %>%
        get_base_plot()+
        geom_junction(data = junctions, junction.y.max = 1/(xmax-xmin)) +
        # geom_junction(data = junctions, junction.y.max = 0.5, color = "#006eff") +
        coord_cartesian(xlim = c(xmin, xmax)) +
        labs(title = intron_group) +
        theme_light()+
        # theme(legend.position = "none") +
        geom_text(
            data = add_exon_number(filter(annotation, type == "exon"), "transcript_name"),
            aes(x = (start + end) / 2, label = exon_number),
            size = 3.5,
            nudge_y = 0.4
            )
}