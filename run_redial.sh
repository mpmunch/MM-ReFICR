#!/bin/bash

#SBATCH --job-name=redial_eval
#SBATCH --output=redial_eval.out
#SBATCH --error=redial_eval.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Run script in container
singularity exec --nv   --env HF_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/huggingface --env TORCH_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/torch   --env PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"   --env TMPDIR=/ceph/project/P9-ReFICR/ReFICR/tmp   p9-reficr_latest.sif   python inference_ReRICR.py --config config/Conv2Item/redial_config.yaml
# singularity exec --nv   --env HF_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/huggingface --env TORCH_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/torch   --env PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"   --env TMPDIR=/ceph/project/P9-ReFICR/ReFICR/tmp   p9-reficr_latest.sif   python inference_ReRICR.py --config config/Conv2Conv/redial_config.yaml
# singularity exec --nv   --env HF_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/huggingface --env TORCH_HOME=/ceph/project/P9-ReFICR/ReFICR/.cache/torch   --env PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"   --env TMPDIR=/ceph/project/P9-ReFICR/ReFICR/tmp   p9-reficr_latest.sif   python inference_ReRICR.py --config config/Ranking/redial_config.yaml
