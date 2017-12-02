#!/bin/bash

#SBATCH
#SBATCH --job-name=FeatureExtractor
#SBATCH --time=10:0:0
#SBATCH --partition=shared
#SBATCH --mail-user=mpeven@jhu.edu
#SBATCH --mem=20G
#SBATCH --array=1,3,5
####SBATCH --array=100-56900:100%50

# Load all Modules
module load anaconda-python/3.6
module load opencv/gcc/3.1.0
module

# Install pip requirements
pip install --user -r requirements.txt

# Run the code
# Change 2 to 100
python ntu_rgb.py $SLURM_ARRAY_TASK_ID 2
echo "Finished with job $SLURM_JOBID"
