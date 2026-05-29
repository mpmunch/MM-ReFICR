#!/bin/bash
# eval_pipeline_local.sh
# Non-Singularity version of eval_pipeline.sh. Runs inference directly
# using the local Python environment (.venv).
#
# Usage:
#   bash eval_pipeline_local.sh <model> [dataset] [from_step]
#
#   model     : linear_NN | dynamic_NN | concat
#                 (e.g. linear_02, dynamic_07, concat)
#   dataset   : inspired (default) | redial
#   from_step : conv2item (default) | conv2conv | ranking | response_gen
#
# Examples:
#   bash eval_pipeline_local.sh linear_02
#   bash eval_pipeline_local.sh dynamic_07 redial
#   bash eval_pipeline_local.sh concat inspired conv2conv
#   bash eval_pipeline_local.sh linear_02 inspired response_gen

cd /work/ReFICR || { echo "Error: cannot cd to /work/ReFICR"; exit 1; }
source .venv/bin/activate
export HF_HOME=./.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
MODEL_ARG="${1:-}"
DATASET="${2:-inspired}"
FROM_STEP="${3:-conv2item}"

if [[ -z "$MODEL_ARG" ]]; then
    echo "Usage: bash eval_pipeline_local.sh <model> [dataset] [from_step]"
    echo "  model : linear_NN | dynamic_NN | concat  (e.g. linear_02, dynamic_07, concat)"
    exit 1
fi

if [[ ! "$MODEL_ARG" =~ ^(linear|dynamic)_[0-9]{2}$|^concat$ ]]; then
    echo "Unknown model: $MODEL_ARG (expected: linear_NN, dynamic_NN, or concat)"
    exit 1
fi

if [[ "$DATASET" != "inspired" && "$DATASET" != "redial" ]]; then
    echo "Unknown dataset: $DATASET (expected: inspired or redial)"
    exit 1
fi

if [[ "$FROM_STEP" != "conv2item" && "$FROM_STEP" != "conv2conv" && "$FROM_STEP" != "ranking" && "$FROM_STEP" != "response_gen" ]]; then
    echo "Unknown from_step: $FROM_STEP (expected: conv2item, conv2conv, ranking, or response_gen)"
    exit 1
fi

MODEL_PATH="model_weights/ReFICR_qlora_${MODEL_ARG}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: model directory not found: ${MODEL_PATH}"
    exit 1
fi

if [[ ! -f "${MODEL_PATH}/adapter_config.json" ]]; then
    echo "Error: adapter_config.json not found in ${MODEL_PATH}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR="training/CRS_data/${DATASET}"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/eval_${DATASET}_${TIMESTAMP}.log"

ITEM_EMB="${DATA_DIR}/${DATASET}_item_embeddings.pt"
CONV_EMB="${DATA_DIR}/${DATASET}_conv_embeddings.pt"
CAND_JSON="${DATA_DIR}/test_processed_cand.jsonl"
CAND_RAG_JSON="${DATA_DIR}/test_processed_cand_rag.jsonl"

# Persistent metric caches — survive between partial runs so the summary
# can display earlier-step metrics even when those steps are skipped.
METRICS_CACHE_CONV2ITEM="${LOG_DIR}/metrics_${DATASET}_conv2item.tmp"
METRICS_CACHE_RANKING="${LOG_DIR}/metrics_${DATASET}_ranking.tmp"

# ---------------------------------------------------------------------------
# Helper: run a step and return its exit code
# ---------------------------------------------------------------------------
run_step() {
    python inference_ReRICR.py --config "$1" --target_model_path "$MODEL_PATH"
}

# ---------------------------------------------------------------------------
# Helper: print a section banner
# ---------------------------------------------------------------------------
banner() {
    echo ""
    echo "======================================================================"
    printf "  %s\n" "$@"
    echo "======================================================================"
}

# ---------------------------------------------------------------------------
# Helper: elapsed time in H:M:S
# ---------------------------------------------------------------------------
elapsed() {
    local secs=$(( $(date +%s) - $1 ))
    printf "%02d:%02d:%02d" $((secs/3600)) $(( (secs%3600)/60 )) $((secs%60))
}

# ---------------------------------------------------------------------------
# Main pipeline (everything below is tee'd to the log file)
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
_STATUS_FILE="${LOG_DIR}/pipeline_status_${DATASET}_${TIMESTAMP}.tmp"

{
    PIPELINE_START=$(date +%s)

    banner \
        "ReFICR Evaluation Pipeline (local)" \
        "Dataset    : ${DATASET}" \
        "From step  : ${FROM_STEP}" \
        "Python     : $(which python)" \
        "Model      : ${MODEL_PATH}" \
        "Log        : ${LOG_FILE}" \
        "Started    : $(date)"

    # -----------------------------------------------------------------------
    # Step 0: Remove stale files
    # -----------------------------------------------------------------------
    banner "CLEANUP — Removing stale files for steps starting from: ${FROM_STEP}"

    case "$FROM_STEP" in
        conv2item)
            STALE=("$ITEM_EMB" "$CONV_EMB" "$CAND_JSON" "$CAND_RAG_JSON")
            ;;
        conv2conv)
            STALE=("$CONV_EMB" "$CAND_RAG_JSON")
            ;;
        ranking)
            STALE=()
            echo "  Resuming from Ranking — no files removed."
            ;;
        response_gen)
            STALE=()
            echo "  Resuming from Response Generation — no files removed."
            ;;
    esac

    for f in "${STALE[@]}"; do
        if [ -f "$f" ]; then
            rm "$f"
            echo "  Removed : $f"
        else
            echo "  Missing (skip) : $f"
        fi
    done

    # -----------------------------------------------------------------------
    # Step 1: Conv2Item
    # -----------------------------------------------------------------------
    STEP1_OK=true
    if [[ "$FROM_STEP" == "conv2item" ]]; then
        banner "[STEP 1/3] Conv2Item — Item Retrieval" "Started : $(date)"
        STEP_START=$(date +%s)

        STEP_TMP="${LOG_DIR}/step1_${DATASET}_${TIMESTAMP}.tmp"
        run_step "config/Conv2Item/${DATASET}_config.yaml" 2>&1 | tee "$STEP_TMP"
        if [ "${PIPESTATUS[0]}" -eq 0 ]; then
            grep -E "Recall@|NDCG@|MRR@" "$STEP_TMP" > "$METRICS_CACHE_CONV2ITEM" 2>/dev/null || true
            echo ""
            echo "  [STEP 1/3] Finished in $(elapsed $STEP_START) — $(date)"
        else
            echo ""
            echo "  [STEP 1/3] FAILED after $(elapsed $STEP_START) — $(date)"
            STEP1_OK=false
            > "$METRICS_CACHE_CONV2ITEM"
        fi
        rm -f "$STEP_TMP"
    else
        banner "[STEP 1/3] Conv2Item — Skipped (resuming from ${FROM_STEP})"
    fi

    # -----------------------------------------------------------------------
    # Step 2: Conv2Conv
    # -----------------------------------------------------------------------
    STEP2_OK=true
    if [[ "$FROM_STEP" == "ranking" || "$FROM_STEP" == "response_gen" ]]; then
        banner "[STEP 2/3] Conv2Conv — Skipped (resuming from ${FROM_STEP})"
    elif [ "$STEP1_OK" = false ]; then
        banner "[STEP 2/3] Conv2Conv — Skipped (Conv2Item failed)"
        STEP2_OK=false
    else
        banner "[STEP 2/3] Conv2Conv — Conversation Retrieval" "Started : $(date)"
        STEP_START=$(date +%s)

        if run_step "config/Conv2Conv/${DATASET}_config.yaml"; then
            echo ""
            echo "  [STEP 2/3] Finished in $(elapsed $STEP_START) — $(date)"
        else
            echo ""
            echo "  [STEP 2/3] FAILED after $(elapsed $STEP_START) — $(date)"
            STEP2_OK=false
        fi
    fi

    # -----------------------------------------------------------------------
    # Step 3: Ranking
    # -----------------------------------------------------------------------
    STEP3_OK=true
    if [[ "$FROM_STEP" == "response_gen" ]]; then
        banner "[STEP 3/3] Ranking — Skipped (resuming from ${FROM_STEP})"
    elif [ "$STEP2_OK" = false ]; then
        banner "[STEP 3/3] Ranking — Skipped (Conv2Conv failed)"
        STEP3_OK=false
        > "$METRICS_CACHE_RANKING"
    else
        banner "[STEP 3/3] Ranking — Item Re-ranking" "Started : $(date)"
        STEP_START=$(date +%s)

        STEP_TMP="${LOG_DIR}/step3_${DATASET}_${TIMESTAMP}.tmp"
        run_step "config/Ranking/${DATASET}_config.yaml" 2>&1 | tee "$STEP_TMP"
        if [ "${PIPESTATUS[0]}" -eq 0 ]; then
            grep -E "Recall@|NDCG@|MRR@" "$STEP_TMP" > "$METRICS_CACHE_RANKING" 2>/dev/null || true
            echo ""
            echo "  [STEP 3/3] Finished in $(elapsed $STEP_START) — $(date)"
        else
            echo ""
            echo "  [STEP 3/3] FAILED after $(elapsed $STEP_START) — $(date)"
            STEP3_OK=false
            > "$METRICS_CACHE_RANKING"
        fi
        rm -f "$STEP_TMP"
    fi

    # -----------------------------------------------------------------------
    # Metric summary
    # -----------------------------------------------------------------------
    banner "METRIC SUMMARY" "Total time : $(elapsed $PIPELINE_START)" "Completed  : $(date)"

    echo ""
    echo "  Conv2Item (retrieval recall before re-ranking):"
    if [ -f "$METRICS_CACHE_CONV2ITEM" ] && [ -s "$METRICS_CACHE_CONV2ITEM" ]; then
        sed 's/^/    /' "$METRICS_CACHE_CONV2ITEM"
        if [[ "$FROM_STEP" != "conv2item" ]]; then
            echo "    (cached from previous run)"
        fi
    else
        echo "    (no metrics — step was skipped and no cache found)"
    fi

    echo ""
    echo "  Ranking (recall after re-ranking):"
    if [ -f "$METRICS_CACHE_RANKING" ] && [ -s "$METRICS_CACHE_RANKING" ]; then
        sed 's/^/    /' "$METRICS_CACHE_RANKING"
        if [ "$STEP3_OK" != true ]; then
            echo "    (cached from previous run)"
        fi
    else
        echo "    (no metrics — step did not run or failed)"
    fi

    echo ""
    echo "  Step status:"
    printf "    Conv2Item : %s\n" "$( [ "$STEP1_OK" = true ] && echo OK || echo FAILED )"
    printf "    Conv2Conv : %s\n" "$( [ "$STEP2_OK" = true ] && echo OK || echo FAILED )"
    printf "    Ranking   : %s\n" "$( [ "$STEP3_OK" = true ] && echo OK || echo FAILED )"
    echo ""

    declare -p STEP1_OK STEP2_OK STEP3_OK > "$_STATUS_FILE"

} 2>&1 | tee "$LOG_FILE"
source "$_STATUS_FILE"
rm -f "$_STATUS_FILE"

echo ""
echo "Full log saved to: $LOG_FILE"

TO_JSON="${MODEL_PATH}/test_processed_gen.jsonl"

{
    banner "[STEP 4/4] Response Generation" "Started : $(date)" "Output  : ${TO_JSON}"
    STEP_START=$(date +%s)

    python inference_ReRICR.py --config "config/Response_Gen/${DATASET}_config.yaml" --target_model_path "$MODEL_PATH" --to_json "$TO_JSON" > /dev/null
    STEP4_STATUS=$?
    STEP4_OK=true
    echo ""
    if [ "$STEP4_STATUS" -eq 0 ]; then
        echo "  [STEP 4/4] Finished in $(elapsed $STEP_START) — $(date)"
    else
        echo "  [STEP 4/4] FAILED after $(elapsed $STEP_START) — $(date)"
        STEP4_OK=false
    fi
} 2>&1 | tee -a "$LOG_FILE"

[[ -f .env ]] && source .env
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "Error: WANDB_API_KEY not set — create .env with it"; exit 1; }

# Log to wandb
WANDB_PROJECT="UCloud Evaluation Pipeline"

RUN_NAME="eval_${DATASET}_$(basename "$MODEL_PATH")_ucloud_${TIMESTAMP}"

python scripts/log_eval_to_wandb.py \
  --project "$WANDB_PROJECT" \
  --run_name "$RUN_NAME" \
  --dataset "$DATASET" \
  --from_step "$FROM_STEP" \
  --model_path "$MODEL_PATH" \
  --conv2item_file "$METRICS_CACHE_CONV2ITEM" \
  --ranking_file "$METRICS_CACHE_RANKING" \
  --log_file "$LOG_FILE" \
  --response_gen_file "$TO_JSON" \
  --step1_ok "$STEP1_OK" \
  --step2_ok "$STEP2_OK" \
  --step3_ok "$STEP3_OK" \
  --step4_ok "$STEP4_OK"
