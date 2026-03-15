#!/bin/bash
#SBATCH --job-name=setup-sahas
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=sahas/logs/%x-%j.out
#SBATCH --error=sahas/logs/%x-%j.err

set -e
mkdir -p ~/logs

# Use /tmp for build artifacts
export TMPDIR=/tmp/$USER-setup
mkdir -p $TMPDIR

echo "=== System info ==="
nvidia-smi
python3 --version

# Check if miniforge is already installed
if [ -d ~/miniforge3 ]; then
    echo "=== Miniforge already installed, skipping ==="
    source ~/miniforge3/bin/activate
else
    echo "=== Installing Miniforge (user-space conda, no sudo needed) ==="
    curl -fsSL -o $TMPDIR/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash $TMPDIR/miniforge.sh -b -p ~/miniforge3
    source ~/miniforge3/bin/activate
fi

echo "=== Creating conda environment 'sahas_bench' ==="
conda create -n sahas_bench python=3.11 -y
conda activate sahas_bench

echo "=== Installing PyTorch ==="
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
echo "Detected CUDA: $CUDA_VER"

case "$CUDA_VER" in
  13.*)            PIP_INDEX="cu126" ; echo "CUDA 13.x detected, using cu126 (closest available)" ;;
  12.[4-9]*|12.1*) PIP_INDEX="cu124" ;;
  12.[1-3]*)       PIP_INDEX="cu121" ;;
  11.8*)           PIP_INDEX="cu118" ;;
  *)               PIP_INDEX="cu126" ; echo "WARNING: Unrecognized CUDA $CUDA_VER, trying cu126" ;;
esac

export PIP_CACHE_DIR=$TMPDIR/pip-cache
mkdir -p $PIP_CACHE_DIR

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/$PIP_INDEX

echo "=== Installing SGLang inference server ==="
pip install "sglang[all]"

echo "=== Installing AIPerf ==="
pip install aiperf

echo "=== Installing LangChain packages ==="
pip install langgraph langchain-openai langchain-core

echo "=== Downloading Qwen3.5-122B-A10B-FP8 model weights ==="
echo "WARNING: This is a ~122GB model and will take significant time to download"
pip install huggingface_hub
export HF_HOME=~/hf-cache
mkdir -p $HF_HOME
huggingface-cli download Qwen/Qwen3.5-122B-A10B-FP8

echo "=== Verifying ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import sglang; print('SGLang OK')"
python -c "import langgraph; print('LangGraph OK')"
python -c "import langchain_openai; print('LangChain-OpenAI OK')"
python -c "import langchain_core; print('LangChain-Core OK')"
aiperf --help | head -3

# Clean up temp files
rm -rf $TMPDIR

echo "=== Setup complete ==="
echo "Your environment name: sahas_bench"
echo ""
echo "To activate in future scripts, add these two lines:"
echo "  source ~/miniforge3/bin/activate"
echo "  conda activate sahas_bench"
