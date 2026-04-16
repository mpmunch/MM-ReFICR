# Create virtual environment
srun singularity exec /ceph/container/python/python_3.10.sif python -m venv --system-site-packages my_venv


# Install packages (example: openpyxl)
srun singularity exec --nv \
     -B my_venv:/scratch/my_venv \
     -B $HOME/.singularity:/scratch/singularity \
     /ceph/container/python/python_3.10.sif \
     /bin/bash -c "export TMPDIR=/scratch/singularity/tmp && \
                   source /scratch/my_venv/bin/activate && \
                   pip install -r requirements.txt --no-cache-dir"