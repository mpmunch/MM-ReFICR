#!/bin/bash

#SBATCH --job-name=reficr_training
#SBATCH --output=logs/reficr_training_%j.out
#SBATCH --error=logs/reficr_training_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

CONTAINER="/ceph/project/rtm-p10/containers/p9-reficr_latest.sif"

# Run script in container
# singularity exec --nv /ceph/project/python/python_3.10.sif bash run.sh
singularity exec --nv --bind /ceph:/ceph $CONTAINER bash run_multi-gpu.sh "$@"
