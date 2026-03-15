#!/bin/bash
#SBATCH --job-name=tb-patterns
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
mkdir -p ~/sahas/logs
source ~/miniforge3/bin/activate
conda activate bench
export HF_HOME=~/hf-cache
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
export SGLANG_DISABLE_CUDNN_CHECK=1

# === Start SGLang server ===
echo "=== Starting SGLang server on port 25000 ==="
python -m sglang.launch_server \
  --model Qwen/Qwen3.5-27B \
  --tp 1 \
  --port 25000 \
  --disable-cuda-graph &
SERVER_PID=$!

echo "=== Waiting for server ==="
for i in $(seq 1 60); do
  if curl -s http://localhost:25000/health > /dev/null 2>&1; then
    echo "Server ready after ~$((i*5))s"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "ERROR: Server did not start within 5 minutes"
    kill $SERVER_PID 2>/dev/null
    exit 1
  fi
  sleep 5
done

# === Run patterns benchmark ===
cd ~/sahas
echo "=== Running testbench patterns ==="
# python -m testbench patterns --test all --n-agents 5 --n-tasks 20
# python -m testbench patterns --test headline --patterns select,json
python -m testbench patterns --test scaling --n-tasks 5 --task-multiplier 5 --patterns cot_select


# === Cleanup ===
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "=== Done ==="
