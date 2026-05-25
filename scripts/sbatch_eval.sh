#!/bin/bash
# Usage: bash scripts/sbatch_eval.sh <weight_id> [from_step]
#   weight_id  : suffix after ReFICR_qlora_ (e.g. 09)
#   from_step  : conv2item (default) | conv2conv | ranking
#
# Submits eval_pipeline.sh for both inspired and redial datasets.

WEIGHT_ID="${1:-}"
FROM_STEP="${2:-conv2item}"

if [[ -z "$WEIGHT_ID" ]]; then
    echo "Usage: bash scripts/sbatch_eval.sh <weight_id> [from_step]"
    echo "  Example: bash scripts/sbatch_eval.sh 09"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${REPO_ROOT}/model_weights/ReFICR_qlora_${WEIGHT_ID}"

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
sbatch "$PIPELINE_SCRIPT" "$MODEL_PATH" inspired "$FROM_STEP"
sbatch "$PIPELINE_SCRIPT" "$MODEL_PATH" redial  "$FROM_STEP"
