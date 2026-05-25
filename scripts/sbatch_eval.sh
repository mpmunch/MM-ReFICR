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

MODEL_PATH="model_weights/ReFICR_qlora_${WEIGHT_ID}"

echo "Submitting eval jobs for: ${MODEL_PATH}"
sbatch scripts/eval_pipeline.sh "$MODEL_PATH" inspired "$FROM_STEP"
sbatch scripts/eval_pipeline.sh "$MODEL_PATH" redial  "$FROM_STEP"
