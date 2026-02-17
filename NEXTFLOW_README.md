# Patch-seq Splicing Analysis - Nextflow Pipeline

This is a Nextflow translation of the original Snakemake pipeline for patch-seq single-cell RNA-seq analysis with focus on splicing events.

## Pipeline Overview

The pipeline performs the following steps:

1. **CREATE_SAMPLE_SHEET** - Generates a sample sheet from the dataset
2. **STAR_INDEX** - Creates STAR genome index (cached, runs once)
3. **STAR_ALIGN** - Aligns RNA-seq reads to the reference genome
4. **TABS_TO_ADATA** - Converts STAR output files to AnnData format
5. **PREPROCESS** - Preprocessing and QC of AnnData objects
6. **GENERATE_BAM_LIST** - Generates lists of BAM files for merging by cell type
7. **MERGE_BAMS** - Merges BAM files by cell type
8. **INDEX_BAMS** - Creates indices for merged BAM files

## Installation

### Prerequisites

- Nextflow (>=21.10.0)
- Conda or Docker/Singularity for containerization
- STAR aligner
- samtools
- Python 3 with required packages (scanpy, pandas, numpy, etc.)

### Setup

```bash
# Install Nextflow
curl -s https://get.nextflow.io | bash
chmod +x nextflow

# Clone the repository
git clone <repository-url>
cd patch_seq_spl

# Create conda environment (optional)
conda env create -f environment.yml
conda activate patch_seq_spl
```

## Configuration

### Environment Variables

Set the following environment variables before running:

```bash
export GENOMIC_DATA_DIR=/path/to/genomic/reference/data
```

This should contain:
- `Ensembl/Mouse/Release_110/Raw/Mus_musculus.GRCm39.dna.primary_assembly.fa`
- `Ensembl/Mouse/Release_110/Raw/Mus_musculus.GRCm39.110.gtf`

### Nextflow Parameters

Key parameters can be passed via command line or in a params file:

```bash
# Command line
nextflow run main.nf --dataset sorensen --group_by three

# Using params file
nextflow run main.nf -params-file params.yaml
```

### Available Profiles

- **standard** - Local execution (default)
- **slurm** - SLURM cluster
- **niagara** - Niagara cluster (requires account setup)
- **docker** - Docker containerization
- **singularity** - Singularity containerization

## Usage

### Basic Run

```bash
export GENOMIC_DATA_DIR=/path/to/genomic/data
nextflow run main.nf --dataset sorensen --group_by three
```

### With SLURM

```bash
export GENOMIC_DATA_DIR=/path/to/genomic/data
nextflow run main.nf -profile slurm --dataset sorensen
```

### With Docker

```bash
export GENOMIC_DATA_DIR=/path/to/genomic/data
nextflow run main.nf -profile docker --dataset sorensen
```

### Resume Pipeline

```bash
nextflow run main.nf -resume
```

### Generate DAG

```bash
nextflow run main.nf -with-dag pipeline_dag.html
```

## Output Structure

```
proc/
├── star/
│   └── {dataset}/
│       ├── {sample}.Aligned.sortedByCoord.out.bam
│       ├── {sample}.SJ.out.tab
│       └── {sample}.ReadsPerGene.out.tab
├── scquint/
│   ├── adata_{group_by}.h5ad
│   └── preprocessed_adata_{group_by}.h5ad
└── merge_bams/
    ├── {cell_type}.bam
    └── {cell_type}.bam.bai

logs/
├── execution_report.html
├── timeline.html
├── trace.txt
└── dag.html
```

## Key Differences from Snakemake

1. **Modularity** - Rules are organized as separate process modules in the `modules/` directory
2. **Channel Operations** - Nextflow uses channels for data flow instead of Snakemake's wildcards
3. **Configuration** - All resource allocation and profiles are in `nextflow.config`
4. **Caching** - STAR_INDEX uses Nextflow's caching mechanism instead of Snakemake's implicit caching
5. **Parallelization** - Nextflow handles process parallelization through its scheduler

## Troubleshooting

### GENOMIC_DATA_DIR not set

```bash
# Check if environment variable is set
echo $GENOMIC_DATA_DIR

# Set it if needed
export GENOMIC_DATA_DIR=/path/to/reference/data
```

### Out of memory errors

Adjust memory allocations in `nextflow.config`:

```groovy
process {
    withName: STAR_ALIGN {
        memory = '64 GB'  // Increase as needed
    }
}
```

### Check pipeline status

```bash
# View running processes
nextflow log

# View last run details
nextflow log -last
```

## Customization

### Modifying Process Parameters

Edit the relevant process block in `nextflow.config`:

```groovy
process {
    withName: YOUR_PROCESS_NAME {
        cpus = 16
        memory = '64 GB'
        time = '24h'
    }
}
```

### Adding New Processes

1. Create a new `.nf` file in `modules/`
2. Define the process with inputs/outputs
3. Include it in `main.nf`
4. Add to workflow

## Additional Information

- [Nextflow Documentation](https://www.nextflow.io/docs/latest/index.html)
- [Nextflow Patterns](https://nextflow-io.github.io/patterns/index.html)
- [STAR Aligner Documentation](https://github.com/alexdobin/STAR)

## License

Same as original project

## Support

For issues or questions, please open an issue in the repository.
