library(Gviz)
library(GenomicFeatures)
library(dplyr)
library(rtracklayer)
library(reticulate)

options(ucscChromosomeNames=FALSE)

anndata <- import("anndata")
adata <- anndata$read_h5ad("proc/scquint/preprocessed_adata_three.h5ad")
sig_intron_attr <- adata$var

gtf_path <- "proc/Mus_musculus.GRCm39.110.gtf"
txdb <- makeTxDbFromGFF(gtf_path, format = "gtf")
txdb <- keepSeqlevels(txdb, c(as.character(c(1:19)), "X", "Y", "MT"))


intronsByTranscripts(txdb)

intron_group <- "Scn1a_2_66181571_-"
plot_coverage_sashimi(intron_group) {
    sig_intron_attr_subset <- sig_intron_attr %>% 
        filter(intron_group == {{intron_group}})
    chromosome <- sig_intron_attr_subset %>% pull(chromosome) %>% unique() %>% as.character()
}

afrom <- 66181571
ato <- 66271179
txTr <- GeneRegionTrack(txdb, chromosome = "2", start = afrom,  end = ato, transcriptAnnotation = "feature", geneSymbols = TRUE)
attributes(txTr)

subset(txTr@range, txTr@range$feature == "CDS")$symbol %>% unique()
txTr@range$symbol %>% unique()


alTrack <- AlignmentsTrack("proc/merge_bams/Lamp5_Lsp1.bam", isPaired = TRUE, start = afrom, end = ato, type = c("coverage", "sashimi"))

plotTracks(
    c(txTr, alTrack),
    from = afrom, to = ato, 
    sashimiFilterTolerance = 5L,
    sashimiFilter = introns,
    sashimiHeight = 0.2,
    sizes = c(1, 1))

GRList <- transcriptsBy(txdb, by = "gene")
aTrack <- AnnotationTrack(GRList$ENSMUSG00000005980)

# Use rtracklayer
annotation_from_gtf <- rtracklayer::import(gtf_path)
annotation_from_gtf <- keepSeqlevels(annotation_from_gtf, c(as.character(c(1:19)), "X", "Y", "MT"), pruning.mode="coarse")
annotation_from_gtf <- annotation_from_gtf[!is.na(mcols(annotation_from_gtf)$gene_name)]
annotation_from_gtf <- subset(annotation_from_gtf, type %in% c("exon", "CDS"))
annotation_from_gtf <- split(annotation_from_gtf, annotation_from_gtf$gene_name)

annotation_from_gtf$Scn1a

plotTracks(aTrack)
