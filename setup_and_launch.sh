#!/bin/bash
# export HF_ENDPOINT=https://hf-mirror.com
# Exit immediately if a command exits with a non-zero status
set -e
# Step 1: Create and activate conda environment
echo "Creating conda environment..."
conda create -n bingpt python=3.8 -y
. $(conda info --base)/etc/profile.d/conda.sh
conda activate bingpt

# Step 2: Install required packages
echo "Installing required packages..."
pip install accelerate==0.33.0 torchvision==0.19.1 webdataset omegaconf einops wandb

# Step 3: Download ImageNet dataset (webdataset format)
echo "Downloading ImageNet dataset..."
huggingface-cli download --repo-type dataset --resume-download Cylarus/ImageNet --local-dir ./datasets/imagenet

# Step 4: Download pretrained tokenizer
echo "Downloading pretrained tokenizer..."
huggingface-cli download orres/btok_w_entropy_reg_0.77 --local-dir ./tokenizer

# Step 5: Confif accelerate
echo "Configuring accelerate (non-interactive)..."
cat <<EOT > ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: 0,1,2,3,4,5,6,7
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOT

# Step 6: Update configuration paths
echo "Updating configuration paths..."
DATASET_PATH="./datasets/imagenet"
TOKENIZER_PATH="tokenizer/vqvae-new_decoder-266k"
CONFIG_FILE="config/gpt_bin.yaml"
sed -i "s|DATASET_PATH|$DATASET_PATH|g" $CONFIG_FILE
sed -i "s|TOKENIZER_PATH|$TOKENIZER_PATH|g" $CONFIG_FILE

# Step 7: Login wandb for monitoring the training
echo "Logging into Weights & Biases..."
wandb login 96131f8aede9a09cdcdaecc19c054f804e330d3d

# Step 7: Launch the training
echo "Launching training..."
sh gpt_train.sh

echo "Training complete!"