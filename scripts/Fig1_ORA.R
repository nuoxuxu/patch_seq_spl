library(msigdbr)
library(fgsea)
library(reticulate)
library(stringr)
library(dplyr)
library(tidyr)
library(pheatmap)
library(jsonlite)
library(ggplot2)
library(rlang)

get_sig_gene_list <- function(glm_results, predictor) {
    predictor_sym <- sym(predictor)
    glm_results %>%
        dplyr::filter(!!predictor_sym < 0.05) %>% 
        # get rownames
        rownames() %>% 
        # extract gene names
        stringr::str_extract("^[^_]*")
}

anndata <- import("anndata")
adata <- anndata$read_h5ad("results/preprocessed_adata_three.h5ad")
glm_results <- adata$uns$simple
prop_labels <- fromJSON("data/mappings/prop_names.json") %>% unlist()
all_predictors <- glm_results %>% colnames()
ephys_props <- all_predictors[!all_predictors %in% c("Lamp5", "Pvalb", "Sncg", "Vip", "Sst", "cpm", "subclass", "soma_depth")]

universe <- glm_results %>%
    rownames() %>%
    stringr::str_extract("^[^_]*") %>%
    unique()

gene_list <- sapply(ephys_props, function(x) {get_sig_gene_list(glm_results, x)})

get_ora_per_subcategory <- function(gene_list, universe, subcategory, rank_by = "count", top = 20) {
    db <- msigdbr(species = "Mus musculus", category = "C5", subcategory = str_glue("GO:{subcategory}"))
    conv <- unique(db[, c("gene_symbol", "gs_exact_source")])
    term_list <- split(x = conv$gene_symbol, f = conv$gs_exact_source)

    fgsea_ora <- lapply(seq_along(gene_list), function(i) {
        fora(pathways = term_list,
            genes = gene_list[[i]], # genes in cluster i
            universe = universe, # all genes
            minSize = 15,
            maxSize = 500) %>%
        mutate(cluster = names(gene_list)[i]) # add cluster column
        }) %>%
        data.table::rbindlist() %>% # combine tables
        filter(padj < 0.05) %>%
        arrange(cluster, padj) %>%
        # Add additional columns from db
        left_join(distinct(db, gs_subcat, gs_exact_source, gs_name, gs_description), by = c("pathway" = "gs_exact_source")) %>%
        # Reformat descriptions
        mutate(gs_name = sub({str_glue("^GO{subcategory}_")}, "", gs_name),
                gs_name = gsub("_", " ", gs_name),
                enrich_score = -log10(pval)) %>%
        select(c("gs_name", "cluster", "enrich_score")) %>%
        pivot_wider(names_from = cluster, values_from = enrich_score)

    if (rank_by == "count") {
        fgsea_ora <- fgsea_ora %>%
            rowwise() %>%
            mutate(count = sum(c_across(where(is.numeric)) > 1.3, na.rm = TRUE)) %>%
            arrange(desc(count)) %>%
            select(-count)
    } else {
        fgsea_ora <- fgsea_ora %>%
            arrange(desc(.data[[rank_by]]))
    }

    if (dim(fgsea_ora)[1] < top) {
        top <- dim(fgsea_ora)[1]
    }
    fgsea_ora <- fgsea_ora[c(1:top), ]
    fgsea_ora$subcategory <- subcategory
    fgsea_ora
}

get_mat_for_heatmap <- function(gene_list, universe, rank_by, top) {
    subcategory_res_list <- lapply(list("MF", "CC", "BP"), function(x) get_ora_per_subcategory(gene_list, universe, x, rank_by, top))
    subcategory_res <- data.table::rbindlist(subcategory_res_list, fill=TRUE)
    mat <- as.matrix(subcategory_res[, -c("gs_name", "subcategory")])
    rownames(mat) <- tolower(subcategory_res$gs_name)
    list(mat, subcategory_res$subcategory)
}

plot_pheatmap <- function(gene_list, universe, rank_by, top) {
    temp <- get_mat_for_heatmap(gene_list, universe, rank_by, top)
    annotation_row <- data.frame(row.names = row.names(temp[[1]]), subcategory =  temp[[2]])
    pheatmap(
        temp[[1]],
        cluster_rows = FALSE,
        cluster_cols = FALSE,
        annotation_row = annotation_row,
        labels_col = prop_labels[colnames(temp[[1]])],
        legend_labels = "-log10(padj)",
        width = 10,
        height = 12,
        # filename = "results/figures/ORA_temp.png"
        )
}

# Use pheatmap for three_multiple
plot_pheatmap(gene_list, universe, "count", 20)

three_simple <- get_mat_for_heatmap(gene_list, universe, "Pvalb", 20)
annotation_row <- data.frame(row.names = row.names(three_simple[[1]]), subcategory =  three_simple[[2]])

pheatmap(
    three_simple[[1]],
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    annotation_row = annotation_row,
    labels_col = colnames(three_simple[[2]]),
    legend_labels = "-log10(padj)",
    width = 7,
    height = 7.5
    )
