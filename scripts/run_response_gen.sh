#!/bin/bash
# run_response_gen.sh
# Runs Response Generation (step 4) for all specified models and datasets.
# No wandb, no CSV — just inference output files.
#
# Usage:
#   bash run_response_gen.sh [output_dir] [dataset] [model...]
#
#   output_dir : where to save results (default: response_gen_results)
#   dataset    : inspired | redial | all (default: all)
#   model...   : one or more model suffixes (e.g. linear_02 concat)
#                if omitted, all model_weights/ReFICR_qlora_* dirs are used
#
# Examples:
#   bash run_response_gen.sh
#   bash run_response_gen.sh response_gen_results all linear_02 concat
#   bash run_response_gen.sh response_gen_results inspired linear_07

cd /work/ReFICR || { echo "Error: cannot cd to /work/ReFICR"; exit 1; }
source .venv/bin/activate
export HF_HOME=./.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
OUTPUT_DIR="${1:-response_gen_results}"
DATASET_ARG="${2:-all}"

# Remaining args are model names; if none given, discover from model_weights/
shift 2 2>/dev/null || true
if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=()
    for d in model_weights/ReFICR_qlora_*/; do
        [[ -f "${d}adapter_config.json" ]] || continue
        suffix="${d#model_weights/ReFICR_qlora_}"
        suffix="${suffix%/}"
        MODELS+=("$suffix")
    done
    if [[ ${#MODELS[@]} -eq 0 ]]; then
        echo "Error: no model_weights/ReFICR_qlora_* directories found"
        exit 1
    fi
fi

if [[ "$DATASET_ARG" == "all" ]]; then
    DATASETS=(inspired redial)
elif [[ "$DATASET_ARG" == "inspired" || "$DATASET_ARG" == "redial" ]]; then
    DATASETS=("$DATASET_ARG")
else
    echo "Unknown dataset: $DATASET_ARG (expected: inspired, redial, or all)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
elapsed() {
    local secs=$(( $(date +%s) - $1 ))
    printf "%02d:%02d:%02d" $((secs/3600)) $(( (secs%3600)/60 )) $((secs%60))
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
TOTAL_START=$(date +%s)
PASS=0
FAIL=0
FAILURES=()

echo ""
echo "======================================================================"
echo "  Response Generation batch run"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Models     : ${MODELS[*]}"
echo "  Datasets   : ${DATASETS[*]}"
echo "  Started    : $(date)"
echo "======================================================================"

for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="model_weights/ReFICR_qlora_${MODEL}"
    if [[ ! -f "${MODEL_PATH}/adapter_config.json" ]]; then
        echo ""
        echo "  [SKIP] ${MODEL} — adapter_config.json not found in ${MODEL_PATH}"
        continue
    fi

    for DATASET in "${DATASETS[@]}"; do
        OUT_DIR="${OUTPUT_DIR}/${MODEL}/${DATASET}"
        mkdir -p "$OUT_DIR"
        TO_JSON="${OUT_DIR}/test_processed_gen.jsonl"

        echo ""
        if [[ -f "$TO_JSON" ]]; then
            echo "  [SKIP] ${MODEL} / ${DATASET} — output already exists: ${TO_JSON}"
            (( PASS++ ))
            continue
        fi

        echo "  [RUN] ${MODEL} / ${DATASET} → ${TO_JSON}"
        STEP_START=$(date +%s)

        python inference_ReRICR.py \
            --config "config/Response_Gen/${DATASET}_config.yaml" \
            --target_model_path "$MODEL_PATH" \
            --to_json "$TO_JSON"

        if [[ $? -eq 0 ]]; then
            echo "  [OK]  ${MODEL} / ${DATASET} — $(elapsed $STEP_START)"
            (( PASS++ ))
        else
            echo "  [FAIL] ${MODEL} / ${DATASET} — $(elapsed $STEP_START)"
            (( FAIL++ ))
            FAILURES+=("${MODEL}/${DATASET}")
        fi
    done
done

echo ""
echo "======================================================================"
echo "  Done — total time: $(elapsed $TOTAL_START)"
echo "  Passed : ${PASS}"
echo "  Failed : ${FAIL}"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "  Failures:"
    for f in "${FAILURES[@]}"; do
        echo "    - $f"
    done
fi
echo "======================================================================"
