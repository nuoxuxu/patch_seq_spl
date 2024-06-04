library(ggtranscript)
library(magrittr)
library(dplyr)

plot_intron_group <- function(sig_intron_attr, transcript_name, intron_group, xmin = NULL, xmax = NULL) {
    gene_of_interest <- strsplit(intron_group, "_")[[1]]
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
}