#!/bin/bash

#SBATCH
#SBATCH --job-name=ImageExtractor
#SBATCH --time=10:0:0
#SBATCH --partition=shared
#SBATCH --mail-user=mpeven@jhu.edu
#SBATCH --mem=25G
#SBATCH --array=1-56800:100

# array (first index)-(last index):(amount to skip)%(amount to run at one time)

# Load virtual environment
source activate activity_recognition

# Run the code
python /home-3/mpeven1\@jhu.edu/work/dev_mp/ntu_rgb/save_images.py $SLURM_ARRAY_TASK_ID
echo "Finished with job $SLURM_JOBID"
