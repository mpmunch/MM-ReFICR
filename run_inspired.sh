#!/bin/bash

#SBATCH --job-name=inspired_eval
#SBATCH --output=inspired_eval.out
#SBATCH --error=inspired_eval.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

CONTAINER="/ceph/project/rtm-p10/containers/p9-reficr_latest.sif"

# Run script in container
# singularity exec --nv /ceph/project/python/python_3.10.sif bash run.sh
singularity exec --nv   --env HF_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/huggingface --env TORCH_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/torch   --env PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"   --env TMPDIR=/ceph/project/P9-ReFICR/ReFICR/tmp   $CONTAINER   python inference_ReRICR.py --config config/Conv2Item/inspired_config.yaml
# singularity exec --nv   --env HF_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/huggingface --env TORCH_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/torch   --env PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"   --env TMPDIR=/ceph/project/P9-ReFICR/ReFICR/tmp   p9-reficr_latest.sif   python inference_ReRICR.py --config config/Conv2Conv/inspired_config.yaml
# singularity exec --nv   --env HF_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/huggingface --env TORCH_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/torch   --env PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"   --env TMPDIR=/ceph/project/P9-ReFICR/ReFICR/tmp   p9-reficr_latest.sif   python inference_ReRICR.py --config config/Ranking/inspired_config.yaml
# singularity exec --nv   --env HF_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/huggingface --env TORCH_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/torch   --env PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"   --env TMPDIR=/ceph/project/P9-ReFICR/ReFICR/tmp   p9-reficr_latest.sif   python inference_ReRICR.py --config config/Dialoge_Manage/inspired_config.yaml
# singularity exec --nv   --env HF_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/huggingface --env TORCH_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/torch   --env PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"   --env TMPDIR=/ceph/project/P9-ReFICR/ReFICR/tmp   p9-reficr_latest.sif   python inference_ReRICR.py --config config/Response_Gen/inspired_config.yaml