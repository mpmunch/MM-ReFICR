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


    

# Copy the rest of the application code into the container
COPY config/ ./config
COPY training/ ./training
COPY inference_ReRICR.py .
COPY ReFICR.py .
COPY requirements.txt .
COPY run.sh .
COPY utils.py .



# Creates a new user called reficr-user and sets it as the user to run the container - this is very nice for security :))
RUN groupadd -r reficr-group && useradd -r -g reficr-group reficr-user

RUN chown -R reficr-user:reficr-group /app

USER reficr-user

EXPOSE 5000

CMD ["sh", "run.sh"]
