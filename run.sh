export HF_HOME=./.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Patch transformers with custom modeling_mistral.py (bidirectional attention)
TRANSFORMERS_PATH=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
cp modeling_mistral.py "$TRANSFORMERS_PATH/models/mistral/modeling_mistral.py"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 25900\
 -m training.run \
 --output_dir model_weights/ReFICR_qlora\
 --model_name_or_path GritLM/GritLM-7B \
 --train_data training/toy_data_instruct/ReFICR_Instruct\
 --learning_rate 2e-5 \
 --num_train_epochs 1 \
 --warmup_ratio 0.03 \
 --per_device_train_batch_size 1 \
 --gradient_accumulation_steps 2 \
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
 --attn_implementation sdpa \
 --pooling_method mean \
 --gradient_checkpointing True \
 --save_strategy "steps" \
 --save_steps 500 \
 --bf16 True \
 --qlora True \
 --report_to none \
 --in_batch_neg False
