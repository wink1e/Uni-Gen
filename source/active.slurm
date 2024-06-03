#!/bin/bash
#SBATCH -J python_active
#SBATCH -N -1
#SBATCH -p cnall
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=56

source ~/WORK/libin/miniconda3/condabin/conda
conda init bash
conda activate unigen
python active.py
