#!/bin/bash

#SBATCH --job-name=reficr_training
#SBATCH --output=logs/reficr_training_%j.out
#SBATCH --error=logs/reficr_training_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00


mkdir -p logs

srun singularity exec --nv \
     -B /ceph/project/rtm-p10:/ceph/project/rtm-p10 \
     -B my_venv:/scratch/my_venv \
     /ceph/container/python/python_3.10.sif \
     /bin/bash -c 'source /scratch/my_venv/bin/activate && exec "$@"' _ \
     bash scripts/run_multi-gpu.sh "$@"
