library(shiny)
library(here)
library(dplyr)
library(tidyr)
library(plotly)
library(glue)
library(ggtranscript)
library(jsonlite)
library(SingleCellExperiment)

# Read in adata and convert to SingleCellExperiment
# use_condaenv("/scratch/s/shreejoy/nxu/CIHR/envs")
# anndata <- import("anndata")
# count <- anndata$read_h5ad("data/interactive_heatmap.h5ad")
# count_sce <- zellkonverter::AnnData2SCE(count)
# saveRDS(count_sce, "data/count_sce.rds")
# setwd("patch_spl_app")
# create_gene_plot("Esrrg_1_187525521_+", "v_baseline", count_sce)
# function to create gene scatterplot
create_gene_plot <- function(intron_group_test, efeature_test, count_sce) {
    intron_idx <- rowData(count_sce) %>% as.data.frame() %>% filter(intron_group == intron_group_test) %>% rownames() %>% as.numeric()
    intron_idx <- intron_idx + 1
    intron_arr <- assay(count_sce)[intron_idx, ]
    cells_to_use <- colSums(intron_arr) > 0
    intron_arr <- intron_arr[, cells_to_use]

    ephys_vector <- reducedDims(count_sce)[[efeature_test]][cells_to_use]

    actual_psi <- sweep(intron_arr, 2, colSums(intron_arr), "/") %>% t() %>% as.matrix() %>% as.data.frame()
    actual_psi$type <- "observed"
    actual_psi <- cbind(actual_psi, ephys_vector)
    
    predicted_psi <- assays(count_sce)[[efeature_test]][intron_idx, cells_to_use] %>% t() %>% as.matrix() %>% as.data.frame()
    predicted_psi$type <- "predicted"
    predicted_psi <- cbind(predicted_psi, ephys_vector)

    df_for_scatter_plot <- rbind(actual_psi, predicted_psi) %>%
        pivot_longer(cols = -c(ephys_vector, type), names_to = "intron_idx", values_to = "psi")

    plot <- df_for_scatter_plot %>%
        ggplot(aes(x = psi, y = ephys_vector, colour = type)) +
        geom_point(size = 1, alpha = 0.5) +
        facet_wrap(~intron_idx, scales = "free") +
        labs(
            title = sprintf("%s vs. %s", intron_group_test, props_labels[[efeature_test]]),
            x = "observed PSI", 
            y = props_labels[[efeature_test]])
    return(plot)
}


# function to to create ggtranscript plot

# plot_intron_group(intron_group_test, sig_intron_attr, annotation_from_gtf)

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

plot_intron_group <- function(intron_group, sig_intron_attr, annotation_from_gtf, xmin = NULL, xmax = NULL) {
    gene_of_interest <- strsplit(intron_group, "_")[[1]][[1]]
    annotation <- annotation_from_gtf %>%
        filter(gene_name == gene_of_interest)
    transcript_name <- annotation %>% pull(transcript_name) %>% unique()
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
        filter(intron_group == {{intron_group}})
    junctions$transcript_name <- transcript_name
    g <- annotation %>%
        get_base_plot() +
            geom_junction(data = junctions, junction.y.max = 0.5) +
            scale_color_manual(values = c("True" = "red", "False" = "blue")) +
            coord_cartesian(xlim = c(xmin, xmax)) +
            labs(title = intron_group) +
            theme_light()+
            theme(legend.position = "none")
            # geom_text(
            #     data = add_exon_number(filter(annotation, type == "exon"), "transcript_name"),
            #     aes(x = (start + end) / 2, label = exon_number),
            #     size = 3.5,
            #     nudge_y = 0.4
            #     )
    g <- g + ylab(NULL) + xlab(NULL)
    return(g)
}

# UI and server

ui <- fluidPage(
  fluidRow(
    # column(6,
    #        selectInput("select", h3("Subclass as a covariate or not?"),
    #                    choices = list("No subclass" = 1, "Subclass" = 2), selected = 1, height = "100px")),
    column(6, plotlyOutput("heat", height = "1200px")),
    column(6, plotlyOutput("scatterplot", height = "500px")),
    column(6, plotOutput("ggtranscript_plot", click = "plot_click", height = "200px"))
  )
)

# read fixed inputs
props_labels <- fromJSON("data/props_labels.json")
annotation_from_gtf <- read.csv("data/annotation_from_gtf.csv")

# reactive paths
path_to_sce <- "data/count_sce.rds"
path_to_corr_matrix <- "data/corr_matrix.csv"

# read reactive inputs
count_sce <- readRDS(path_to_sce)
cor_mat <- read.csv(path_to_corr_matrix, row.names = "intron_group")
intron_group_order <- rownames(cor_mat)
sig_intron_attr <- as.data.frame(rowData(count_sce))
sig_intron_attr <- sig_intron_attr %>% select(c(intron, chromosome, start, end, strand, intron_group, gene_name))
reordered_cor_mat <- as.matrix(cor_mat[rev(intron_group_order), ])

server <- function(input, output, session) {
  observeEvent(input$select, {
    subclass_as_covariate <- input$select
    if (subclass_as_covariate == 1) {
      path_to_sce <- "data/count_sce_subclass.rds"
      path_to_corr_matrix <- "data/corr_matrix_subclass.csv"
    } else {
      path_to_sce <- "data/count_sce.rds"
      path_to_corr_matrix <- "data/corr_matrix.csv"
    }
  })

  output$heat <- renderPlotly({
    plot_ly(source = "heat_plot", colors = "RdPu", reversescale = FALSE) %>%
      add_heatmap(
        x = colnames(reordered_cor_mat),
        y = rownames(reordered_cor_mat),
        z = reordered_cor_mat
      ) %>%
    layout(
        title = "Correlation Heatmap",
        xaxis = list(
            tickmode = "array",
            tickvals = seq(0, length(colnames(reordered_cor_mat)) - 1, 1),
            ticktext = props_labels[colnames(reordered_cor_mat)] %>% unname() %>% unlist()
            ),
        yaxis = list(
            title = "Intron group"
            )
    )
})

  output$scatterplot <- renderPlotly({
      clickData <- event_data("plotly_click", source = "heat_plot")
      if (is.null(clickData)) {
      return(NULL)
      }

      gp <- create_gene_plot(clickData[["y"]], clickData[["x"]], count_sce)
      ggplotly(gp) # Add gp as an argument to ggplotly
  })

  output$ggtranscript_plot <- renderPlot({
      clickData <- event_data("plotly_click", source = "heat_plot")
      if (is.null(clickData)) {
      return(NULL)
      }

      gp <- plot_intron_group(clickData[["y"]], as.data.frame(rowData(count_sce)), annotation_from_gtf)
      gp
  })
}

shinyApp(ui, server)
