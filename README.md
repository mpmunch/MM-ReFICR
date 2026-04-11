# mm-ReFICR
This repo is a Master's project based on creating a multimodal extension on top of the [ReFICR](#reficr) system.

## Setup
### Training Data
You can download the instruction data from the link

https://drive.google.com/file/d/1U_45qCHXpiArW_BCkKOhVfOQn23J4pUA/view?usp=drive_link

place it in the directory: `training/toy_data_instruct/ReFICR_Instruct`


### Patch for bidirectional attention
In order to use the bidirectional attention, you need to replace the `modeling_mistral.py` file in your transformers installation with the patched `setup/modeling_mistral.py` file.

The provided `setup/patch_mistral.sh` script should detect the correct install path and replace the file.

!!Be sure to use in a virtual environment!!

More details can be found in [ContextualAI/gritlm](https://github.com/ContextualAI/gritlm).

### Python Virtual Environment - UV
A virtual environment is recommended for setting up and running the project, if not using the provided `pytorch.Dockerfile`.
Python 3.10 has been used for development, but 3.8 has also been tested to run correctly.
For our project, UV has been used to manage the installed Python versions and environments.

#### Install UV (might require specifying a local install location by using the `--install-dir` flag on some HPCs)
``uv python install 3.10``

#### Creating the virtual environment
``uv venv --python 3.10``
#### Installing requirements
``uv pip install -r requirements.txt``

### Docker
A docker image can be build from the .Dockerfile or pulled from Docker Hub. The local tag is meant to be used for a isolated Docker container, while the `pytorch.Dockerfile` is used for Singularity on the AAU HPC.

To run the Docker container locally (requires the NVIDIA Container Toolkit), with shared memory and unlocked memlock:

``
docker run --rm -it \
--gpus all \
--ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 \
mathiaspm/p9-reficr:local
``

## Scripts
Scripts with the `_local` suffix are meant to be used directly inside a container (in our case, provided by UCloud - SDU eScience Center), while the other scripts use the Slurm workload manager and a custom built Singulrity .sif file (in our case, running on AI-LAB - AAU CLAAUDIA).

### Training
Our ReFICR extension is trained and evaluated on both a multi-GPU setup (NVIDIA L4), and on a single-GPU setup (NVIDIA H100).

#### Multi-GPU
The `run_sbatch.sh` is a Slurm wrapper to call the `run_multi-gpu.sh` in a Singularity container. The script uses a default of 4 GPUs and is tested to run on the NVIDIA L4. The primary modification to the training with multi-GPU is that it uses a batch size of 1 and enabled cross-device negatives.

#### Single-GPU
The `run_local.sh` is, as described, meant to be used directly in a container and changes directory to the `/work/ReFICR` workspace. The script also includes its own basic logging, working independently of any job scheduler. 

### Evaluation
The eval_pipeline scripts calls the scripts defined in [Inference](#inference) and creates cached files to enable partial re-evaluation while keeping old metrics.


## Inference
To run inference on the trained QLoRA, the following scripts can be called. (See [Evaluation](#evaluation))

`Conv2Item` and `Conv2Conv + Ranking` includes Recall Metrics while the `scripts/calc_distn.py` allows to calculate Distinct N after running the `Response Generation` step.
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




## Acknowledgement
### ReFICR
[yt556677/ReFICR](https://github.com/yt556677/ReFICR) The base of our project is built upen the ReFICR project.

### GritLM
[ContextualAI/gritlm](https://github.com/ContextualAI/gritlm) The original repo, and therefore also ours, is built upon GritLM.
