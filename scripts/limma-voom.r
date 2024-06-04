library(edgeR)
library(reticulate)
library(zellkonverter)
library(SingleCellExperiment)
library(magrittr)
library(jsonlite)

# Get cpm adata object and convert it to SingleCellExperiment object
use_condaenv("/scratch/s/shreejoy/nxu/CIHR/envs")
anndata <- import("anndata")
adata <- anndata$read_h5ad("/scratch/s/shreejoy/nxu/CIHR/proc/cpm.h5ad")
adata <- AnnData2SCE(adata)

# Keep only cells that are assigned a subclass
transcriptomic_ID_subclass <- fromJSON("/scratch/s/shreejoy/nxu/patch_seq_spl_archive/proc/mappings/transcriptomic_ID_subclass.json")
keep_cells <- intersect(names(transcriptomic_ID_subclass), colnames(assay(adata, "X")))

# Preprocessing
d0 <- DGEList(assay(adata, "X")[ ,keep_cells])
d0 <- calcNormFactors(d0)

cutoff <- 3
drop <- which(apply(edgeR::cpm(d0), 1, max) < cutoff)
d <- d0[-drop, ]

group <- transcriptomic_ID_subclass[colnames(d0)] %>%
    unlist() %>%
    unname() %>%
    as.factor()

plotMDS(d, col = as.numeric(group))

# Voom transformation and calculation of variance weights
mm <- model.matrix(~0 + group)
y <- voom(d, mm, plot = T)

# plot something in ggplot using car data
library(ggplot2)
library(car)

ggplot(mtcars, aes(x=wt, y=mpg)) + geom_point() + geom_smooth(method="lm")