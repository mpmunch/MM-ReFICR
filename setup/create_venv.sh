#!/bin/bash
set -euo pipefail

# Create virtual environment inside the container at the mount path so pip shebangs resolve correctly
mkdir -p my_venv
mkdir -p "$HOME/.singularity/tmp"
srun singularity exec --nv \
     -B my_venv:/scratch/my_venv \
     /ceph/container/python/python_3.10.sif \
     python -m venv --system-site-packages /scratch/my_venv


# Install packages
srun singularity exec --nv \
     -B /ceph/project/rtm-p10:/ceph/project/rtm-p10 \
     -B my_venv:/scratch/my_venv \
     -B "$HOME/.singularity":/scratch/singularity \
     /ceph/container/python/python_3.10.sif \
     /bin/bash -c "export TMPDIR=/scratch/singularity/tmp && \
                   source /scratch/my_venv/bin/activate && \
                   pip install -r requirements.txt --no-cache-dir"


# Activate the virtual environment and run patch_mistral.sh to patch transformers with custom modeling_mistral.py (bidirectional attention).
srun singularity exec --nv \
     -B /ceph/project/rtm-p10:/ceph/project/rtm-p10 \
     -B my_venv:/scratch/my_venv \
     /ceph/container/python/python_3.10.sif \
     /bin/bash -c "source /scratch/my_venv/bin/activate && \
                   bash setup/patch_mistral.sh"
