# Preparation

## Setting up environment
Install required packages in a new conda environment
```bash
mamba create -n patch_seq_spl numpy pandas scanpy statsmodels anndata pytorch::pytorch pyro-ppl scipy cffi scikit-learn tqdm snakemake r-svglite pyarrow
```


`r-arrow` cannot be installed through conda, for macOS it can be installed:
```r
install.packages("arrow")
```
However, for CentOS it is more complicated, C++17 is required which is not provided by CentOS, see [Installation details](https://arrow.apache.org/docs/r/articles/developers/install_details.html)

TODO Install r-arrow on CentOS Linux
A solution is have one seqparate environment for r-arrow, `test_arrow`
### on local machine
Download processed data from Niagara
```bash
rsync -av nxu@nia-datamover1.scinet.utoronto.ca:/scratch/s/shreejoy/nxu/patch_seq_spl/proc/ proc
```
Tested under macOS
### On a computing cluster
- Install conda environment from environment.yaml
```bash
mamba env create --prefix ./env --file environment.yaml
```
- Use `Python: Select Interpreter command` in VS code to set the workspace-level Python interpreter as `.env/bin/python`
- Turn on `python.terminal.activateEnvInCurrentTerminal`
- Create a symbolic link, absolute path has to be used 
```bash
ln -s /scratch/s/shreejoy/nxu/patch_seq_spl/env ${CONDA_PREFIX}/envs/patch_seq_spl
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
