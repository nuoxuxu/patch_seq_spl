#!/bin/bash
#SBATCH -J CPM_as_predictor
#SBATCH -t 0-24:0
#SBATCH -N 1
#SBATCH -c 80
#SBATCH -o slurm_logs/CPM_as_predictor.log

conda activate ./envs
python scripts/CPM_as_predictor.py
# python scripts/GLM_per_ephys.py rheo_width
# python scripts/GLM_per_ephys.py rheo_upstroke_downstroke_ratio