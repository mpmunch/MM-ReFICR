#!/bin/bash
# cd "$(dirname "$0")"
cd /work/ReFICR

# export PATH="$HOME/ucloud/.local/bin:$PATH"
# export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=./.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
source .env

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "Error: WANDB_API_KEY is not set. Please export it in your environment before running this script." >&2
  exit 1
fi

export WANDB_PROJECT="MM_ReFICR Training"

# ------------------------CHANGE PARAMS HERE!! ------------------------
IMAGE_FUSION_MODE="${1:-linear}"   # Options: linear, concat, or dynamic
IMAGE_FUSION_WEIGHT="${2:-0.2}"

if [[ "${IMAGE_FUSION_MODE}" != "linear" && "${IMAGE_FUSION_MODE}" != "concat" && "${IMAGE_FUSION_MODE}" != "dynamic" ]]; then
  echo "Error: IMAGE_FUSION_MODE must be 'linear', 'concat', or 'dynamic'. Got: ${IMAGE_FUSION_MODE}" >&2
  exit 1
fi

EXTRA_FUSION_ARGS=()
if [[ "${IMAGE_FUSION_MODE}" == "linear" ]]; then
  EXTRA_FUSION_ARGS+=(--image_fusion_weight "${IMAGE_FUSION_WEIGHT}")
  export WANDB_NAME="Train-IFW${IMAGE_FUSION_WEIGHT}"
elif [[ "${IMAGE_FUSION_MODE}" == "dynamic" ]]; then
  EXTRA_FUSION_ARGS+=(--image_fusion_weight "${IMAGE_FUSION_WEIGHT}")
  export WANDB_NAME="Train-dynamic${IMAGE_FUSION_WEIGHT}"
else
  export WANDB_NAME="Train-concat"
fi

OUTPUT_SUFFIX="${IMAGE_FUSION_MODE}"
if [[ "${IMAGE_FUSION_MODE}" == "linear" || "${IMAGE_FUSION_MODE}" == "dynamic" ]]; then
  OUTPUT_SUFFIX="${IMAGE_FUSION_MODE}_${IMAGE_FUSION_WEIGHT}"
fi

# ------------------------------------------------
LOG_DIR="logs/${IMAGE_FUSION_MODE}"
if [[ "${IMAGE_FUSION_MODE}" == "linear" ]]; then
  TAG="${IMAGE_FUSION_WEIGHT#0.}"
  [[ "${TAG}" == "${IMAGE_FUSION_WEIGHT}" ]] && TAG="${IMAGE_FUSION_WEIGHT//./}"
  [[ ${#TAG} -eq 1 ]] && TAG="0${TAG}"
  LOG_DIR="${LOG_DIR}/weight_${TAG}"
fi
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"
ERR_FILE="${LOG_DIR}/run_${TIMESTAMP}.err"

echo "Full log: $LOG_FILE"
echo "Errors/warnings: $ERR_FILE"

# Patch transformers with custom modeling_mistral.py (bidirectional attention) -- Moved into patch_mistral.sh for better modularity.
# TRANSFORMERS_PATH=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
# cp modeling_mistral.py "$TRANSFORMERS_PATH/models/mistral/modeling_mistral.py"

source .venv/bin/activate
echo python interpreter: $(which python)

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 25900\
 -m training.run \
 --output_dir "model_weights/ReFICR_qlora_${OUTPUT_SUFFIX/./}"\
 --model_name_or_path GritLM/GritLM-7B \
 --train_data training/toy_data_instruct/ReFICR_Instruct\
 --learning_rate 2e-5 \
 --num_train_epochs 2 \
 --warmup_ratio 0.03 \
 --per_device_train_batch_size 2 \
 --gradient_accumulation_steps 1 \
 --dataloader_drop_last True \
 --normalized True \
 --temperature 0.02 \
 --query_max_len 512 \
 --passage_max_len 1024 \
 --generative_max_len 2048 \
 --train_group_size 10 \
 --mode unified \
 --lora True \
 --attn bbcc \
 --attn_implementation eager \
 --pooling_method mean \
 --gradient_checkpointing True \
 --save_strategy "epoch" \
 --save_steps 500 \
 --bf16 True \
 --qlora True \
 --report_to wandb \
 --in_batch_neg False \
 --use_image_features True \
 --image_embeddings_path training/CRS_data/posters/inspired_clip_embeddings.pt \
 --image_fusion_mode ${IMAGE_FUSION_MODE} \
 "${EXTRA_FUSION_ARGS[@]}" \
 --run_name "${WANDB_NAME}" \
 2> >(tee "$ERR_FILE" >&2) | tee "$LOG_FILE"
