library(VGAM)
library(magrittr)
library(SingleCellExperiment)
library(dplyr)

# library(reticulate)
# library(zellkonverter)
# use_condaenv("/scratch/s/shreejoy/nxu/CIHR/envs")
# anndata <- import("anndata")
# adata <- anndata$read_h5ad("/scratch/s/shreejoy/nxu/CIHR/proc/adata_filtered.h5ad")
# adata <- AnnData2SCE(adata)
# saveRDS(adata, "/scratch/s/shreejoy/nxu/CIHR/proc/adata_filtered.rds")

adata <- readRDS("/scratch/s/shreejoy/nxu/CIHR/proc/adata_filtered.rds")
Fgf12_idx <- data.frame(rowData(adata)) %>%
    filter(intron_group == "Fgf12_16_28217141_-") %>%
    rownames()
rheo_upstroke_downstroke_ratio_name <- reducedDims(adata)$rheo_upstroke_downstroke_ratio
data_df <- cbind(rheo_upstroke_downstroke_ratio_name, assay(adata, "X")[Fgf12_idx, ] %>% t() %>% as.matrix() %>% as.data.frame())
data_df <- data_df %>% rename(rheo_upstroke_downstroke_ratio = rheo_upstroke_downstroke_ratio_name)
data_df <- data_df[rowSums(data_df[, Fgf12_idx]) != 0, ]
colnames(data_df) <- paste0("X", colnames(data_df))

vgam(cbind(X6654, X6655, X6656, X6657, X6658) ~ Xrheo_upstroke_downstroke_ratio, data = data_df,
     cumulative(parallel = TRUE), trace = TRUE)