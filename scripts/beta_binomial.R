library(lmtest)
library(broom)
library(Matrix)
library(jsonlite)
library(tidyverse)
library(parallel)
library(glue)

args <- commandArgs(trailingOnly = TRUE)
predictor <- args[1]

transcriptomic_ID_subclass <- fromJSON("data/mappings/transcriptomic_ID_subclass.json") %>% unlist()
event_names <- readLines("data/quantas/event_names.txt")
sample_names <- readLines("data/quantas/sample_names.txt")
ephys_data <- read.csv("data/ephys_data_sc.csv", row.names = 1)

isoform1Tags <- readMM("data/quantas/isoform1Tags.mtx")
colnames(isoform1Tags) <- event_names
rownames(isoform1Tags) <- sample_names

isoform2Tags <- readMM("data/quantas/isoform2Tags.mtx")
colnames(isoform2Tags) <- event_names
rownames(isoform2Tags) <- sample_names

# keep_columns <- !(colSums(isoform1Tags) == 0) & !(colSums(isoform2Tags) == 0)
keep_columns <- colnames(isoform1Tags)[(colSums(isoform1Tags) + colSums(isoform2Tags)) > 250]
isoform1Tags <- isoform1Tags[, keep_columns]
isoform2Tags <- isoform2Tags[, keep_columns]

run_regression <- function(event_name, prop_name) {
    temp <- data.frame(isoform1Tags = isoform1Tags[, event_name], isoform2Tags = isoform2Tags[, event_name])
    temp$n <- temp$isoform1Tags + temp$isoform2Tags
    temp$ephys_prop <- ephys_data[rownames(temp), prop_name]
    temp <- temp[!is.na(temp$ephys_prop) & (temp$n != 0), ]
    model0 <- glm(isoform1Tags/n ~ 1, data = temp, family = binomial, weights = n)
    model1 <- glm(isoform1Tags/n ~ ephys_prop, data = temp, family = binomial, weights = n)
    statistic <- tidy(model1)[1, "statistic"] %>% pull()
    p_value <- lrtest(model0, model1) %>%
        tidy() %>%
        pull(p.value) %>%
        .[[2]]
    c(statistic, p_value)
}

out <- mclapply(keep_columns, function(x) {run_regression(x, predictor)}, mc.cores = 12)
out <- do.call(rbind, out)
colnames(out) <- c("statistic", "p_value")
out <- as.data.frame(out)
out$event_name <- keep_columns
write.csv(out, glue("results/quantas/{predictor}.csv"), row.names = FALSE)