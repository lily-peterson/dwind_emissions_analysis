#!/bin/bash

#SBATCH --account=dwind
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --qos=high

source ~/miniconda3/etc/profile.d/conda.sh  # or wherever conda.sh lives
conda activate dwind

python wetryagain.py