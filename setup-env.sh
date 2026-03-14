#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
 
set -e
mkdir -p ~/logs
 
# Use /tmp for build artifacts
export TMPDIR=/tmp/$USER-setup
mkdir -p $TMPDIR
 
echo "=== System info ==="
nvidia-smi
python3 --version
 
echo "=== Installing Miniforge (user-space conda, no sudo needed) ==="
curl -fsSL -o $TMPDIR/miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash $TMPDIR/miniforge.sh -b -p ~/miniforge3
source ~/miniforge3/bin/activate
 
echo "=== Creating conda environment ==="
conda create -n bench python=3.11 -y
conda activate bench
 
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
 
echo "=== Installing inference servers ==="
pip install "sglang[all]"
pip install vllm
 
echo "=== Installing AIPerf ==="
pip install aiperf
 
echo "=== Downloading model weights ==="
pip install huggingface_hub
export HF_HOME=~/hf-cache
mkdir -p $HF_HOME
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
 
echo "=== Verifying ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import sglang; print('SGLang OK')"
python -c "import vllm; print('vLLM OK')"
aiperf --help | head -3
 
# Clean up temp files
rm -rf $TMPDIR
 
echo "=== Setup complete ==="
echo "To activate in future scripts, add these two lines:"
echo "  source ~/miniforge3/bin/activate"
echo "  conda activate bench"
