#!/bin/bash
#SBATCH --job-name=download-model
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p ~/logs
source ~/miniforge3/bin/activate
conda activate bench
export HF_HOME=~/hf-cache

echo "Downloading Qwen3.5-27B..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-27B', cache_dir='$HF_HOME')
print('Download complete!')
"
