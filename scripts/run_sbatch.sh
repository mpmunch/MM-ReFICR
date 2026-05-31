#!/bin/bash

#SBATCH --job-name=reficr_training
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00


cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

mkdir -p logs

MODE="${1:-linear}"
WEIGHT="${2:-0.0}"
LOG_DIR="logs/${MODE}"
if [[ "${MODE}" == "linear" || "${MODE}" == "dynamic" ]]; then
     TAG="${WEIGHT#0.}"
     [[ "${TAG}" == "${WEIGHT}" ]] && TAG="${WEIGHT//./}"
     [[ ${#TAG} -eq 1 ]] && TAG="0${TAG}"
     LOG_DIR="${LOG_DIR}/weight_${TAG}"
fi
mkdir -p "${LOG_DIR}"

srun \
     --output="${LOG_DIR}/training_${SLURM_JOB_ID}.out" \
     --error="${LOG_DIR}/training_${SLURM_JOB_ID}.err" \
     singularity exec --nv \
     -B /ceph/project/rtm-p10:/ceph/project/rtm-p10 \
     -B my_venv:/scratch/my_venv \
     /ceph/container/python/python_3.10.sif \
     /bin/bash -c 'source /scratch/my_venv/bin/activate && exec "$@"' _ \
     bash scripts/run_multi-gpu.sh "$@"
