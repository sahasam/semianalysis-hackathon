#!/bin/bash
#SBATCH --job-name=download-qwen
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

echo "=== Job started at $(date) ==="
echo "=== Working directory: $(pwd) ==="
echo "=== Home directory: $HOME ==="

echo "=== Initializing conda for this shell ==="
# This is the critical part that's often missing
eval "$(~/miniforge3/bin/conda shell.bash hook)"

echo "=== Conda initialized ==="
conda info

echo "=== Activating sahas_bench environment ==="
conda activate sahas_bench

echo "=== Environment activated ==="
conda info --envs
echo "Active environment: $CONDA_DEFAULT_ENV"

echo "=== Verifying tools ==="
which python
which pip
which huggingface-cli
python --version
pip --version
huggingface-cli --version

echo "=== Downloading Qwen3.5-122B-A10B-FP8 model ==="
echo "This is ~122GB and will take 1-2 hours..."
export HF_HOME=~/hf-cache
mkdir -p $HF_HOME

huggingface-cli download Qwen/Qwen3.5-122B-A10B-FP8

echo "=== Verifying download ==="
ls -lh ~/hf-cache/
du -sh ~/hf-cache/

echo "=== Job completed at $(date) ==="
