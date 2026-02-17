#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import modules
include { CREATE_SAMPLE_SHEET_GOUWENS } from './modules/create_sample_sheet'
include { CREATE_SAMPLE_SHEET_SORENSEN } from './modules/create_sample_sheet'
// include { STAR_INDEX } from './modules/star_index'
// include { STAR_ALIGN } from './modules/star_align'
// include { TABS_TO_ADATA } from './modules/tabs_to_adata'
// include { PREPROCESS } from './modules/preprocess'
// include { GENERATE_BAM_LIST } from './modules/generate_bam_list'
// include { MERGE_BAMS } from './modules/merge_bams'
// include { INDEX_BAMS } from './modules/index_bams'

// Parse parameters
params.metadata_gouwens = "data/gouwens/20200711_patchseq_metadata_mouse.csv"
params.manifest_gouwens = "data/gouwens/2021-09-13_mouse_file_manifest.csv"
params.metadata_sorensen = "data/sorensen/pseq_genotype_reporter_by_id.csv"
params.manifest_sorensen = "data/sorensen/exc_patchseq_file_manifest.csv"
params.group_by = "three"
params.genomic_data_dir = "/project/rrg-shreejoy/Genomic_references/"

workflow {
    // Step 1: Create sample sheet
    CREATE_SAMPLE_SHEET_GOUWENS("gouwens", file(params.metadata_gouwens), file(params.manifest_gouwens))
    
    // // Step 2: Create STAR index (if not already exists)
    // star_index = STAR_INDEX(
    //     file("${params.genomic_data_dir}/Ensembl/Mouse/Release_110/Raw/Mus_musculus.GRCm39.dna.primary_assembly.fa"),
    //     file("${params.genomic_data_dir}/Ensembl/Mouse/Release_110/Raw/Mus_musculus.GRCm39.110.gtf")
    // )
    
    // // Step 3: Prepare input channel for STAR alignment
    // samples = sample_sheet
    //     .splitCsv(header: true)
    //     .map { row -> 
    //         [row.sample, row.dataset] 
    //     }
    
    // // Step 4: STAR alignment
    // star_bams = STAR_ALIGN(samples, star_index)
    
    // // Step 5: Convert tabs to AnnData
    // adata = TABS_TO_ADATA(sample_sheet, params.group_by)
    
    // // Step 6: Preprocess
    // preprocessed_adata = PREPROCESS(adata)
    
    // // Step 7: Generate BAM lists and merge BAMs
    // cell_types = star_bams
    //     .map { sample, dataset, bam, sj, counts -> sample }
    //     .distinct()
    
    // bam_lists = GENERATE_BAM_LIST(cell_types)
    // merged_bams = MERGE_BAMS(bam_lists)
    // indexed_bams = INDEX_BAMS(merged_bams)
}
