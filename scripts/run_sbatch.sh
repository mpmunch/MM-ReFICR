#!/bin/bash

#SBATCH --job-name=reficr_training
#SBATCH --output=reficr_training.out
#SBATCH --error=reficr_training.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

# Run script in container
# singularity exec --nv /ceph/project/python/python_3.10.sif bash run.sh
singularity exec --nv p9-reficr_latest.sif bash run_multi-gpu.sh
