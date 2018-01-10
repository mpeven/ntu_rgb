#!/bin/bash

#SBATCH
#SBATCH --job-name=FeatureExtractor
#SBATCH --time=10:0:0
#SBATCH --partition=shared
#SBATCH --mail-user=mpeven@jhu.edu
#SBATCH --mem=20G
#SBATCH --array=2,4,6
####SBATCH --array=100-56900:100%50

# Load virtual environment
source activate activity_recognition

# Run the code
# Change 2 to 100
python ntu_rgb.py $SLURM_ARRAY_TASK_ID 2
echo "Finished with job $SLURM_JOBID"
