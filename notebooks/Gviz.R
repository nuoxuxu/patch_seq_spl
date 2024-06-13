suppressMessages(
    {
        library(Gviz)
        library(GenomicFeatures)
        library(dplyr)
        library(rtracklayer)
        library(stringr)
    }
)

plot_Gviz <- function(my_intron_group, bam_vec, transcripts_subset, fill_coverage_vec) {
    stopifnot(length(bam_vec) == length(fill_coverage_vec))
    
    sig_intron_attr_subset <- subset(sig_intron_attr, mcols(sig_intron_attr)$intron_group == my_intron_group)
    exonByTranscript <- get_exonByTranscript(my_intron_group)

    xlim <- get_xlim(sig_intron_attr_subset, exonByTranscript)
    afrom <- xlim[1]
    ato <- xlim[2]

    my_chromosome <- seqnames(sig_intron_attr_subset) %>% unique() %>% as.character()
    
    exonByTranscript <- exonByTranscript[transcripts_subset]

    atrack <- AnnotationTrack(unlist(exonByTranscript), chromosome = my_chromosome, shape = "box", 
        group=as.factor(mcols(unlist(exonByTranscript))$transcript_name))
    
    bam <- bam_vec[1]
    alTrack_1 <- AlignmentsTrack(glue("proc/merge_bams/{bam}.bam"), 
        isPaired = TRUE, start = afrom, end = ato, type = c("coverage", "sashimi"), 
        chromosome = my_chromosome, fill.coverage = fill_coverage_vec[1], name = str_replace_all(bam, "_", " "))
    
    bam <- bam_vec[2]
    alTrack_2 <- AlignmentsTrack(glue("proc/merge_bams/{bam}.bam"), 
        isPaired = TRUE, start = afrom, end = ato, type = c("coverage", "sashimi"),
        chromosome = my_chromosome, fill.coverage = fill_coverage_vec[2], name = str_replace_all(bam, "_", " "))

    plotTracks(
        c(alTrack_1, alTrack_2, atrack), from = afrom, to = ato, sashimiHeight = 0.1, 
        sashimiFilter = introns, main = my_intron_group, groupAnnotation = "group", size = c(1, 1, 10), just.group = "above")
}

# Load data and config
options(ucscChromosomeNames=FALSE)
source("scripts/transcript_viz.r")
sig_intron_attr <- get_sig_intron_attr()
introns <- read.csv("proc/scquint/sig_intron_attr.csv") %>%
    makeGRangesFromDataFrame(keep.extra.columns = TRUE)

# Start plotting here
my_intron_group <- "Cacna1a_8_85306098_+"
bam_vec <- c("Pvalb_Vipr2", "Sst_Calb2_Pdlim5")
transcripts_subset <- c("Cacna1a-213")
color_vec <- c("red", "blue")

pdf(glue("proc/figures/Gviz_{my_intron_group}.pdf"))
plot_Gviz(my_intron_group, bam_vec, transcripts_subset, color_vec)
dev.off()