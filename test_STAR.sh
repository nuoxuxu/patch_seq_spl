#!/bin/bash

STAR --runThreadN 8 \
     --genomeDir /project/rrg-shreejoy/Genomic_references/Ensembl/Mouse/Release_110/STAR_index \
     --readFilesIn /scratch/nxu/patch_seq_spl/data/sorensen/transcriptome/SM-GE4WX_E1-50_GATAGATCAG-ATGGCGTACG/SM-GE4WX_E1-50_GATAGATCAG-ATGGCGTACG.R1.fastq.gz \
     --readFilesCommand zcat \
     --outFileNamePrefix test \
     --outSAMtype BAM SortedByCoordinate
