library(dplyr)
library(monocle3)
library(zellkonverter)
library(reticulate)
library(glue)

use_condaenv("~/miniforge3")
anndata <- import("anndata")
count <- anndata$read_h5ad("proc/adata_count.h5ad")
count <- zellkonverter::AnnData2SCE(count)

gene_names <- read.csv("proc/gene_names_for_monocle3.csv", header = FALSE)

cds <- new_cell_data_set(assays(count)$X,
    cell_metadata = as.data.frame(colnames(count),
        row.names = colnames(count),
        gene_metadata = as.data.frame(count)
    )
)

colData(cds)$rheo_threshold_v <- colData(count)$rheo_threshold_v

cds_subset <- cds[rownames(cds) %in% gene_names$V1,]

# chunksize <- 11442
# iterable_list <- list(c(1:chunksize), c(chunksize:(2 * chunksize)), c((2 * chunksize):(3 * chunksize)), c((3 * chunksize):(4 * chunksize)))

gene_fits <- fit_models(cds_subset, model_formula_str = "~rheo_threshold_v", cores = 80)
fit_coefs <- coefficient_table(gene_fits)
emb_time_terms <- fit_coefs %>% filter(term == "rheo_threshold_v")
inspect <- emb_time_terms %>%
    # filter(q_value < 0.05) %>%
    select(gene_id, q_value, estimate)
write.csv(inspect, glue("proc/monocle3_results.csv"))

compare_models(gene_fits)
