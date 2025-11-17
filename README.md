# ReFICR
## Commands

### srun Commands
srun --mem=24G --cpus-per-task=15 --gres=gpu:1 --time=01:00:00 singularity exec --env HF_HOME=~/.cache/huggingface --nv p9-reficr_latest.sif bash run.sh

`srun --mem=24G --cpus-per-task=15 --gres=gpu:1 --time=01:00:00 singularity exec --nv p9-reficr_latest.sif bash run.sh`

## Dependency
`pip install -r requirements.txt`

### Docker


sudo docker build -t mathiaspm/reficr -f pytorch.Dockerfile .

docker pull mathiaspm/p9-reficr:local

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 5000:5000 \
  mathiaspm/p9-reficr:local

## Training
### Data
You can download the instruction data from the link

https://drive.google.com/file/d/1U_45qCHXpiArW_BCkKOhVfOQn23J4pUA/view?usp=drive_link

place it in the directory (training/toy_data_instruct/ReFICR_Instruct)

### Model Weight Download
We will upload our model weight to huggingface later

`sh run.sh`
```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 25900\
 -m training.run \
 --output_dir model_weights/ReFICR_qlora \
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
 --attn_implementation sdpa \
 --pooling_method mean \
 --gradient_checkpointing True \
 --save_strategy "epoch" \
 --save_steps 500 \
 --bf16 True \
 --qlora True \
 --in_batch_neg False
 ```

## Inference
### Recommendation
#### The performance of retrieved candiate items(Conv2Item)
`CUDA_VISIBLE_DEVICES=0 python inference_ReRICR.py --config config/Conv2Item/inspired_config.yaml`
#### The performance of ranking(Conv2Conv + Ranking)
`CUDA_VISIBLE_DEVICES=0 python inference_ReRICR.py --config config/Conv2Conv/inspired_config.yaml`

`CUDA_VISIBLE_DEVICES=0 python inference_ReRICR.py --config config/Ranking/inspired_config.yaml`
### Dialogue Management
`CUDA_VISIBLE_DEVICES=0 python inference_ReRICR.py --config config/Dialoge_Manage/inspired_config.yaml`
### Response Generation
`CUDA_VISIBLE_DEVICES=0 python inference_ReRICR.py --config config/Response_Gen/inspired_config.yaml`

### For AAU-AILAB

`srun --mem=24G --cpus-per-task=15 --gres=gpu:1 --time=01:00:00 singularity exec --nv   --env HF_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/huggingface --env TORCH_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/torch   --env PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"   --env TMPDIR=/ceph/project/P9-ReFICR/ReFICR/tmp   p9-reficr_latest.sif   python inference_ReRICR.py {ADD CORRECT CONFIG FLAG}`


## Note
you need to simply replace the modeling_mistral.py file in your transformers installation with modeling_mistral.py in order to use the bidirectional attention. More details can be found in [ContextualAI/gritlm](https://github.com/ContextualAI/gritlm).

## Acknowledgement
[yt556677/ReFICR](https://github.com/yt556677/ReFICR) This repo is a student project based on the ReFICR paper.

[ContextualAI/gritlm](https://github.com/ContextualAI/gritlm) The original repo and our is built upon gritlm!
