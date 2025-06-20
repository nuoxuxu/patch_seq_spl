library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(GenomicRanges)
library(GenomicFeatures)
library(rtracklayer)

SJ_names <- read_delim("lampert_data/SJ_names.csv", delim="_")

SJ_names <- SJ_names %>%
    mutate(
        chrom = str_extract(SJ, "^[^:]+"),
        end = str_extract(SJ, "(?<=:)[^_]+"),
        strand = str_extract(SJ, "(?<=_)[^_]+"),
        start = str_extract(SJ, "(?<=_)[^_]+$")
    ) %>% 
    filter(strand %in% c("+", "-", "*"))

SJ_genes <- GRanges(
    seqnames = SJ_names$chrom,
    ranges = IRanges(start = as.numeric(SJ_names$start), end = as.numeric(SJ_names$end)),
    strand = SJ_names$strand
)

pig_genes <- rtracklayer::import("lampert_data/Sus_scrofa.Sscrofa11.1.105.chr.gtf") %>%
    subset(
        type == "gene"
    )

hits <- findOverlaps(SJ_genes, pig_genes)

pig_gene_hits <- bind_cols(as_tibble(hits), as_tibble(pig_genes[subjectHits(hits)]$gene_name)) %>% 
    rename(
       value = "pig_gene_name"
    ) %>%
    dplyr::select(queryHits, pig_gene_name)

SJ_names %>%
    mutate(row_index = row_number()) %>% 
    left_join(
        pig_gene_hits,
        join_by(
            row_index == queryHits
        )
    ) %>%
    drop_na(pig_gene_name) %>%
    distinct(row_index, .keep_all = TRUE) %>% 
    dplyr::select(SJ, pig_gene_name) %>% 
    write_csv("lampert_data/SJ_pig_genes.csv")