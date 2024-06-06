# Preparation

## Setting up environment

### Setting up conda environment
Install required packages in a new conda environment from `environment.yml`.
If you are doing this on a local machine:
```bash
mamba env create -n patch_seq_spl
```
If you are doing this on Niagara, clone the repository to `$SCRATCH` and `cd` to the project root:
```bash
mamba env create -p ./envs/patch_seq_spl
```

- Use `Python: Select Interpreter command` in VS code to set the workspace-level Python interpreter as `.env/bin/python`
- Turn on `python.terminal.activateEnvInCurrentTerminal`
- Create a symbolic link, absolute path has to be used 
```bash
ln -s /scratch/s/shreejoy/nxu/patch_seq_spl/env ${CONDA_PREFIX}/envs/patch_seq_spl
```

### Setting up R environment
Since R packages on conda-forge channels are out-dated in general and the bioconda channel does not support osx-arm64, it's better to not install R packages within the conda environment.

Install R packages required for this project
```r
remotes::install_github("nx10/httpgd")
install.packages(c("BiocManager", "tidyverse", "languageserver", "rlang", "arrow", "devtools", "reticulate", "svglite", "ggvenn", "rtracklayer"))
BiocManager::install(c("GenomicRanges", "GenomeInfoDb", "S4Vectors"))
devtools::install_github("dzhang32/ggtranscript")
```

## Getting data

Download processed data from Niagara

```bash
rsync -av nxu@nia-datamover1.scinet.utoronto.ca:/scratch/s/shreejoy/nxu/patch_seq_spl/proc/ proc
```

## Get genomic references 
All genomic reference files can be found in $GENOMIC_DATA_DIR
```bash
cat > ~/.bashrc <<END
export GENOMIC_DATA_DIR="/project/s/shreejoy/Genomic_references/"
END
```
## Get outputs from STAR
```bash
cp /scratch/s/shreejoy/nxu/patch_seq_spl/data/SJ_out_tabs/* data/SJ_out_tabs/
```
# run Snakemake
Make sure that `.env` is activated. If you followed previous steps, `.env` should be automatically activated in any new terminals you create.
Dry-run
```
snakemake --profile profile/default -np
```

# Olego
## Installation
1. Install pre-compiled Olego from https://sourceforge.net/projects/ngs-olego/files/
2. Create symbolic link to ~/miniforge3/condabin
```bash
ln -s /scratch/s/shreejoy/nxu/patch_seq_spl/tools/olego.bin.linux.x86_64.v1.1.5/olego /scratch/s/shreejoy/nxu/patch_seq_spl/env/bin/
ln -s /scratch/s/shreejoy/nxu/patch_seq_spl/tools/olego.bin.linux.x86_64.v1.1.5/olegoindex /scratch/s/shreejoy/nxu/patch_seq_spl/env/bin/
```

```bash
conda config --env --append channels conda-forge
conda config --env --append channels bioconda
mamba install --yes -c chaolinzhanglab quantas
```

```bash
git clone https://github.com/chaolinzhanglab/czplib
export PERL5LIB=/scratch/s/shreejoy/nxu/patch_seq_spl/tools/czplib
```
## Build the index for the genome sequence
```bash
olegoindex /scratch/s/shreejoy/nxu/Genomic_references/mm39/Mus_musculus.GRCm39.dna.primary_assembly.fa -p proc/
```
