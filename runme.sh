#!/bin/bash

#SBATCH
#SBATCH --job-name=FeatureExtractor
#SBATCH --time=10:0:0
#SBATCH --partition=shared
#SBATCH --mail-user=mpeven@jhu.edu
#SBATCH --mem=25G
#SBATCH --array=1-5690%500

# array (first index)-(last index):(amount to skip)%(amount to run at one time)

# Load virtual environment
source activate activity_recognition

# Run the code
python /home-3/mpeven1\@jhu.edu/work/dev_mp/ntu_rgb/ntu_rgb.py $SLURM_ARRAY_TASK_ID 10
echo "Finished with job $SLURM_JOBID"
