#!/bin/bash
#SBATCH --job-name=setup-env2
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# THIS FILE IS CREATED BECAUSE VLLM AND SGLANG HAVE DEPENDENCY CONFLICTS
# THIS WAS RUN AFTER SETUP-ENV.SH
# THIS ALSO FIXES A SEPARATE ISSUE WHERE HUGGINGFACE REQUIRES A AUTH KEY 
# FOR DOWNLOADING OLLAMA (WTF?) SO QWEN IS BEING USED.

set -e
mkdir -p ~/logs
source ~/miniforge3/bin/activate

# Remove the broken env, start fresh
conda remove -n bench --all -y 2>/dev/null || true
conda create -n bench python=3.11 -y
conda activate bench

export TMPDIR=/tmp/$USER-setup
export PIP_CACHE_DIR=$TMPDIR/pip-cache
mkdir -p $TMPDIR $PIP_CACHE_DIR

echo "=== Installing SGLang (includes PyTorch) ==="
pip install --upgrade pip
pip install "sglang[all]"

echo "=== Installing AIPerf ==="
pip install aiperf

echo "=== Downloading model (ungated, no login needed) ==="
export HF_HOME=~/hf-cache
mkdir -p $HF_HOME
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

echo "=== Verifying ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import sglang; print('SGLang OK')"
aiperf --help | head -3

rm -rf $TMPDIR
echo "=== Setup complete ==="
