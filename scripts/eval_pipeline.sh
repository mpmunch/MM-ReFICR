#!/bin/bash
#SBATCH --job-name=reficr_eval
#SBATCH --output=reficr_eval.out
#SBATCH --error=reficr_eval.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# eval_pipeline.sh
# Runs the Conv2Item → Conv2Conv → Ranking evaluation pipeline in sequence.
# Clears stale embeddings and intermediate files before starting.
# Saves all output and a metric summary to a timestamped log file.
#
# Usage:
#   bash eval_pipeline.sh [target_model_path] [dataset] [from_step]
#   target_model_path : path to the model to evaluate
#   dataset           : inspired (default) or redial
#   from_step         : conv2item (default) | conv2conv | ranking
#               When resuming from a later step, stale files from earlier
#               steps are preserved so they can be reused.
#
# Examples:
#   bash eval_pipeline.sh model_weights/ReFICR_qlora inspired conv2item
#   bash eval_pipeline.sh model_weights/ReFICR_qlora redial conv2item
#   bash eval_pipeline.sh model_weights/ReFICR_qlora inspired conv2conv
#   bash eval_pipeline.sh model_weights/ReFICR_qlora inspired ranking

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
else
    SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
    cd "$(dirname -- "$SCRIPT_PATH")/.." || exit 1
fi

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
TARGET_MODEL_PATH="${1:-}"
DATASET="${2:-inspired}"
FROM_STEP="${3:-conv2item}"

if [[ "$DATASET" != "inspired" && "$DATASET" != "redial" ]]; then
    echo "Unknown dataset: $DATASET (expected: inspired or redial)"
    exit 1
fi

if [[ "$FROM_STEP" != "conv2item" && "$FROM_STEP" != "conv2conv" && "$FROM_STEP" != "ranking" ]]; then
    echo "Unknown from_step: $FROM_STEP (expected: conv2item, conv2conv, or ranking)"
    exit 1
fi

if [[ -z "$TARGET_MODEL_PATH" ]]; then
    echo "ERROR: TARGET_MODEL_PATH not specified!"
    echo "Usage: bash eval_pipeline.sh [target_model_path] [dataset] [from_step]"
    echo "  Example: bash eval_pipeline.sh /path/to/model inspired conv2item"
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
# Singularity settings (cluster)
# ---------------------------------------------------------------------------
CONTAINER="/ceph/container/python/python_3.10.sif"
SING_BINDS=(
    "--bind" "/ceph/project/rtm-p10:/ceph/project/rtm-p10"
    "--bind" "my_venv:/scratch/my_venv"
)
SING_ENVS=(
    "--env" "HF_HOME=${PWD}/.cache/huggingface"
    "--env" "TORCH_HOME=${PWD}/.cache/torch"
    "--env" "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
    "--env" "TMPDIR=${PWD}/tmp"
)

# ---------------------------------------------------------------------------
# Helper: run a step and return its exit code
# ---------------------------------------------------------------------------
run_step() {
    local config="$1"
    singularity exec --nv "${SING_BINDS[@]}" "${SING_ENVS[@]}" "$CONTAINER" \
        /bin/bash -c 'source /scratch/my_venv/bin/activate && exec "$@"' _ \
        python inference_ReRICR.py --config "$config" --target_model_path "$TARGET_MODEL_PATH"
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

{
    PIPELINE_START=$(date +%s)

    banner \
        "ReFICR Evaluation Pipeline" \
        "Dataset    : ${DATASET}" \
        "From step  : ${FROM_STEP}" \
        "Model      : ${TARGET_MODEL_PATH}" \
        "Container  : ${CONTAINER}" \
        "Log        : ${LOG_FILE}" \
        "Started    : $(date)"

    # -----------------------------------------------------------------------
    # Step 0: Remove stale files
    # Only remove files that will be regenerated by the steps we are running.
    # Files produced by skipped steps are kept so they can be reused.
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

        STEP_TMP="${LOG_DIR}/step1_${TIMESTAMP}.tmp"
        run_step "config/Conv2Item/${DATASET}_config.yaml" 2>&1 | tee "$STEP_TMP"
        if [ "${PIPESTATUS[0]}" -eq 0 ]; then
            grep -E "Recall@|NDCG@|MRR@" "$STEP_TMP" > "$METRICS_CACHE_CONV2ITEM" 2>/dev/null || true
            echo ""
            echo "  [STEP 1/3] Finished in $(elapsed $STEP_START) — $(date)"
        else
            echo ""
            echo "  [STEP 1/3] FAILED after $(elapsed $STEP_START) — $(date)"
            STEP1_OK=false
        fi
        rm -f "$STEP_TMP"
    else
        banner "[STEP 1/3] Conv2Item — Skipped (resuming from ${FROM_STEP})"
    fi

    # -----------------------------------------------------------------------
    # Step 2: Conv2Conv
    # -----------------------------------------------------------------------
    STEP2_OK=true
    if [[ "$FROM_STEP" == "ranking" ]]; then
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
    if [ "$STEP2_OK" = false ]; then
        banner "[STEP 3/3] Ranking — Skipped (Conv2Conv failed)"
        STEP3_OK=false
    else
        banner "[STEP 3/3] Ranking — Item Re-ranking" "Started : $(date)"
        STEP_START=$(date +%s)

        STEP_TMP="${LOG_DIR}/step3_${TIMESTAMP}.tmp"
        run_step "config/Ranking/${DATASET}_config.yaml" 2>&1 | tee "$STEP_TMP"
        if [ "${PIPESTATUS[0]}" -eq 0 ]; then
            grep -E "Recall@|NDCG@|MRR@" "$STEP_TMP" > "$METRICS_CACHE_RANKING" 2>/dev/null || true
            echo ""
            echo "  [STEP 3/3] Finished in $(elapsed $STEP_START) — $(date)"
        else
            echo ""
            echo "  [STEP 3/3] FAILED after $(elapsed $STEP_START) — $(date)"
            STEP3_OK=false
        fi
        rm -f "$STEP_TMP"
    fi

    # -----------------------------------------------------------------------
    # Metric summary  (grep recall lines, tag each with its step)
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

} 2>&1 | tee "$LOG_FILE"

TO_JSON="$TARGET_MODEL_PATH/test_processed_gen.jsonl"
singularity exec --nv "${SING_BINDS[@]}" "${SING_ENVS[@]}" "$CONTAINER" \
    /bin/bash -c 'source /scratch/my_venv/bin/activate && exec "$@"' _ \
      python inference_ReRICR.py --config "config/Response_Gen/${DATASET}_config.yaml"  --target_model_path "$TARGET_MODEL_PATH" --to_json "$TO_JSON"  

echo ""
echo "Full log saved to: $LOG_FILE"

  source .env
  # Log to wandb 
  WANDB_PROJECT="MMReFICR Evaluation Pipeline"

  MODEL_PATH="${TARGET_MODEL_PATH}"
  RUN_NAME="eval_${DATASET}_$(basename "$TARGET_MODEL_PATH")"

  singularity exec --nv "${SING_BINDS[@]}" "${SING_ENVS[@]}" "$CONTAINER" \
    /bin/bash -lc "source /scratch/my_venv/bin/activate && python scripts/log_eval_to_wandb.py \
      --project \"$WANDB_PROJECT\" \
      --run_name \"$RUN_NAME\" \
      --dataset \"$DATASET\" \
      --from_step \"$FROM_STEP\" \
      --model_path \"$MODEL_PATH\" \
      --conv2item_file \"$METRICS_CACHE_CONV2ITEM\" \
      --ranking_file \"$METRICS_CACHE_RANKING\" \
      --response_gen_file \"$TO_JSON\" \
      --step1_ok \"$STEP1_OK\" \
      --step2_ok \"$STEP2_OK\" \
      --step3_ok \"$STEP3_OK\""
