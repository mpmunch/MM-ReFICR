#!/bin/bash
set -euo pipefail

# Simple wrapper that runs the alpha-log plotting script on a CPU node via srun.
#
# Usage:
#   bash scripts/plot_alpha_log.sh redial
#   bash scripts/plot_alpha_log.sh inspired
#
# This expects the alpha log to be located at:
#   logs/dynamic/analysis/dynamic_alpha_<dataset>.jsonl
# and will write PNGs into the same folder.

# Resolve repo root (so relative paths work from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/plot_alpha_log.sh <dataset>

Where <dataset> is:
  inspired | redial

Edit the script to change:
  - histogram bin count
  - scatter max points
EOF
}

DATASET="${1:-}"
if [[ -z "${DATASET}" ]]; then
  echo "Error: dataset is required (inspired|redial)" >&2
  usage
  exit 2
fi

if [[ "${DATASET}" != "inspired" && "${DATASET}" != "redial" ]]; then
  echo "Error: unknown dataset '${DATASET}' (expected inspired|redial)" >&2
  usage
  exit 2
fi

# Make Singularity tmp available (mirrors setup/create_venv.sh)
mkdir -p "$HOME/.singularity/tmp"

CONTAINER="/ceph/container/python/python_3.10.sif"

# ------------------- CONFIGURE THESE -------------------
# Number of bins in the alpha histogram
BINS=20
# Max points to plot in the scatter (keeps PNG readable)
MAX_POINTS=20000
# -------------------------------------------------------

LOG_DIR="logs/dynamic/analysis"
INPUT="${LOG_DIR}/dynamic_alpha_${DATASET}.jsonl"

if [[ ! -f "${INPUT}" ]]; then
  echo "Error: input log not found: ${INPUT}" >&2
  echo "Did you run eval with alpha_log_path set for ${DATASET}?" >&2
  exit 1
fi

PLOT_ARGS=("--input" "${INPUT}" "--bins" "${BINS}" "--max_points" "${MAX_POINTS}")

# Run inside container + venv. No --nv (GPU) needed.
srun --cpus-per-task=2 --mem=4G --time=00:10:00 \
  singularity exec \
  -B /ceph/project/rtm-p10:/ceph/project/rtm-p10 \
  -B my_venv:/scratch/my_venv \
  -B "$HOME/.singularity":/scratch/singularity \
  "${CONTAINER}" \
  /bin/bash -lc "export TMPDIR=/scratch/singularity/tmp && source /scratch/my_venv/bin/activate && python scripts/plot_alpha_log.py ${PLOT_ARGS[*]}"
