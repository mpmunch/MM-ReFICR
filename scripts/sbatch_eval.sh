#!/bin/bash
# Usage: bash scripts/sbatch_eval.sh <weight_id> [dataset] [from_step]
#   weight_id  : suffix after ReFICR_qlora_linear_ (e.g. 09)
#   dataset    : inspired | redial | both (default: both)
#   from_step  : conv2item (default) | conv2conv | ranking
#
# Examples:
#   bash scripts/sbatch_eval.sh 02
#   bash scripts/sbatch_eval.sh 02 redial
#   bash scripts/sbatch_eval.sh 02 inspired conv2conv

WEIGHT_ID="${1:-}"
DATASET="${2:-both}"
FROM_STEP="${3:-conv2item}"

if [[ -z "$WEIGHT_ID" ]]; then
    echo "Usage: bash scripts/sbatch_eval.sh <weight_id> [dataset] [from_step]"
    echo "  Examples:"
    echo "    bash scripts/sbatch_eval.sh 02"
    echo "    bash scripts/sbatch_eval.sh 02 redial"
    echo "    bash scripts/sbatch_eval.sh 02 inspired conv2conv"
    exit 1
fi

if [[ "$DATASET" != "inspired" && "$DATASET" != "redial" && "$DATASET" != "both" ]]; then
    echo "Error: dataset must be 'inspired', 'redial', or 'both'. Got: ${DATASET}"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${REPO_ROOT}/model_weights/ReFICR_qlora_linear_${WEIGHT_ID}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: model directory not found: ${MODEL_PATH}"
    exit 1
fi

if [[ ! -f "${MODEL_PATH}/adapter_config.json" ]]; then
    echo "Error: adapter_config.json not found in ${MODEL_PATH}"
    exit 1
fi

PIPELINE_SCRIPT="${REPO_ROOT}/scripts/eval_pipeline.sh"

echo "Submitting eval jobs for: ${MODEL_PATH}"
[[ "$DATASET" == "inspired" || "$DATASET" == "both" ]] && sbatch "$PIPELINE_SCRIPT" "$MODEL_PATH" inspired "$FROM_STEP"
[[ "$DATASET" == "redial"  || "$DATASET" == "both" ]] && sbatch "$PIPELINE_SCRIPT" "$MODEL_PATH" redial  "$FROM_STEP"
