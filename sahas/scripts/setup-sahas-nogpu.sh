#!/bin/bash
#SBATCH --job-name=setup-sahas-nogpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=sahas/logs/%x-%j.out
#SBATCH --error=sahas/logs/%x-%j.err

set -e
mkdir -p ~/logs

# Use /tmp for build artifacts
export TMPDIR=/tmp/$USER-setup
mkdir -p $TMPDIR

echo "=== Installing Miniforge ==="
if [ -d ~/miniforge3 ]; then
    echo "Miniforge already installed, skipping"
    source ~/miniforge3/bin/activate
else
    curl -fsSL -o $TMPDIR/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash $TMPDIR/miniforge.sh -b -p ~/miniforge3
    source ~/miniforge3/bin/activate
fi

echo "=== Creating conda environment 'sahas_bench' ==="
conda create -n sahas_bench python=3.11 -y
conda activate sahas_bench

echo "=== Installing Python packages (CPU versions for now) ==="
export PIP_CACHE_DIR=$TMPDIR/pip-cache
mkdir -p $PIP_CACHE_DIR

pip install --upgrade pip
pip install huggingface_hub
pip install langgraph langchain-openai langchain-core
pip install aiperf

echo "=== Downloading Qwen3.5-122B-A10B-FP8 model weights ==="
echo "This is ~122GB and will take a while..."
export HF_HOME=~/hf-cache
mkdir -p $HF_HOME
huggingface-cli download Qwen/Qwen3.5-122B-A10B-FP8

# Clean up temp files
rm -rf $TMPDIR

echo "=== Phase 1 complete ==="
echo "Next step: Run the GPU setup script to install PyTorch + SGLang"
