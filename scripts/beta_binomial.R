library(lmtest)
library(broom)
library(Matrix)
library(jsonlite)
library(tidyverse)
library(parallel)
library(glue)
library(arrow)
library(VGAM)

# define helper functions
read_isoformTags <- function(file_name) {
    out <- arrow::read_parquet(glue("data/quantas/{file_name}.parquet")) %>% as.data.frame()
    rownames(out) <- out[, "__index_level_0__"]
    dplyr::select(out, -"__index_level_0__")
}

args <- commandArgs(trailingOnly = TRUE)
predictor <- args[1]
statistical_model <- args[2]

# load data
transcriptomic_ID_subclass <- fromJSON("data/mappings/transcriptomic_ID_subclass.json") %>% unlist()
ephys_data <- read.csv("data/ephys_data_sc.csv", row.names = 1)
isoform1Tags <- read_isoformTags("isoform1Tags")
isoform2Tags <- read_isoformTags("isoform2Tags")

# filter for events with at least 250 reads
keep_columns <- colnames(isoform1Tags)[(colSums(isoform1Tags) + colSums(isoform2Tags)) > 250]
isoform1Tags <- isoform1Tags[, keep_columns]
isoform2Tags <- isoform2Tags[, keep_columns]

#TODO: catch error fore cases that give "could not obtain valid initial values" errors
run_regression <- function(isoform1Tags, isoform2Tags, event_name, prop_name, statistical_model) {
    stopifnot(statistical_model == "logistic" | statistical_model == "beta_binomial")

    temp <- data.frame(isoform1Tags = isoform1Tags[, event_name], isoform2Tags = isoform2Tags[, event_name], row.names = rownames(isoform1Tags))
    temp$n <- temp$isoform1Tags + temp$isoform2Tags
    temp$ephys_prop <- ephys_data[rownames(temp), prop_name]
    temp <- temp[!is.na(temp$ephys_prop) & (temp$n != 0), ]
    if (statistical_model == "logistic") {
        model0 <- glm(isoform1Tags/n ~ 1, data = temp, family = binomial, weights = n)
        model1 <- glm(isoform1Tags/n ~ ephys_prop, data = temp, family = binomial, weights = n)
        statistic <- tidy(model1)[1, "statistic"] %>% pull()
        p_value <- lmtest::lrtest(model0, model1) %>%
            tidy() %>%
            pull(p.value) %>%
            .[[2]]
        c(statistic, p_value)
    } else if (statistical_model == "beta_binomial") {
        model0 <- vglm(cbind(isoform1Tags, isoform2Tags) ~ 1, betabinomial(imethod = 1), data = temp)
        model1 <- vglm(cbind(isoform1Tags, isoform2Tags) ~ ephys_prop, betabinomial(imethod = 1), data = temp)
        statistic <- attr(model1, "coefficients")["ephys_prop"] %>% unname()
        p_value <- VGAM::lrtest(model0, model1)@Body[2, "Pr(>Chisq)"]
        c(statistic, p_value)
    }
}

out <- mclapply(keep_columns, function(x) {run_regression(isoform1Tags, isoform2Tags, x, predictor, statistical_model)}, mc.cores = detectCores() - 1)
out <- do.call(rbind, out)
colnames(out) <- c("statistic", "p_value")
out <- as.data.frame(out)
out$event_name <- keep_columns
out <- out[, c("event_name", "statistic", "p_value")]

directory_path <- glue("proc/quantas/{statistical_model}")
if (!dir.exists(directory_path)) {
  dir.create(directory_path, showWarnings = FALSE, recursive = TRUE)
}

write.csv(out, glue("proc/quantas/{statistical_model}/{predictor}.csv"), row.names = FALSE)