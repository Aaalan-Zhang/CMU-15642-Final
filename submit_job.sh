#!/bin/bash
#SBATCH -p GPU-shared         # Partition: RM, RM-shared, GPU, or GPU-shared (adjust as needed)
#SBATCH -t 03:00:00           # Walltime requested (here, 1 hour)             # Number of nodes requested (1 node)
#SBATCH --ntasks-per-node=1   # Number of cores to allocate per node (only relevant for shared partitions)
#SBATCH --gpus=v100-16      # Request 1 GPU of type v100-16 (modify type and count as necessary)
#SBATCH -o myjob_%j.out       # Output file name (%j will be replaced with the job ID)


# Change to the directory from which the job was submitted (optional but recommended)
cd "/jet/home/mxu10/CMU-15642-Final/"
source activate mlsys  # Activate your conda environment (if using conda)

export WANDB_API_KEY=***REMOVED***


# Run your commands
# For example, run a Python script
python train_listmle.py > listmle_${SLURM_JOB_ID}.out 2>&1