# --------- Dockerfile for LOCAL/TESTING USE - NOT FOR AAU AI-LAB!! ---------

# Official PyTorch image (2.1.1) and a compatible CUDA/cuDNN stack.
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Sets the working directory in the container
WORKDIR /app

# --- Set cache directory for huggingface libraries to a writable location ---
# This resolves warnings about the cache folder by placing it inside our app directory.
ENV HF_HOME=/app/.cache/huggingface


COPY requirements.txt .

#This **should** match most of the existing packages in the base image
RUN pip install --no-cache-dir -r requirements.txt



# --- Custom file replacement for bidirectional attention ---
# Instructions from ReFICR GitHub repo
# 1. Copy the custom modeling_mistral.py file into the container's /tmp directory.
COPY modeling_mistral.py /tmp/modeling_mistral.py

# 2. Find the installed transformers package path and overwrite the original file.
RUN TRANSFORMERS_PATH=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))") && \
    cp /tmp/modeling_mistral.py $TRANSFORMERS_PATH/models/mistral/modeling_mistral.py && \
    rm /tmp/modeling_mistral.py

