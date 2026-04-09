export HF_HOME=./.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

torchrun --nproc_per_node 4 --master_port 25900\
 -m training.run \
 --output_dir model_weights/ReFICR_qlora\
 --model_name_or_path GritLM/GritLM-7B \
 --train_data training/toy_data_instruct/ReFICR_Instruct\
 --learning_rate 2e-5 \
 --num_train_epochs 2 \
 --warmup_ratio 0.03 \
 --per_device_train_batch_size 1 \
 --negatives_cross_device True \
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
 --attn_implementation sdpa \
 --pooling_method mean \
 --gradient_checkpointing True \
 --save_strategy "steps" \
 --save_steps 1000 \
 --bf16 True \
 --qlora True \
 --report_to none \
 --in_batch_neg False \
 --use_image_features True \
 --image_embeddings_path training/CRS_data/posters/inspired_clip_embeddings.pt \
 --image_fusion_weight 0.2
