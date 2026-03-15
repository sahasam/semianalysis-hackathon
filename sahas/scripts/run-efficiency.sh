#!/bin/bash
#SBATCH --job-name=tb-efficiency
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
mkdir -p ~/sahas/logs
source ~/miniforge3/bin/activate
conda activate bench
export HF_HOME=~/hf-cache
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
export SGLANG_DISABLE_CUDNN_CHECK=1

# Sweep 3 cache sizes × 6 concurrency levels = 18 data points
for MEM in 0.70 0.80 0.90; do
  echo "=== Starting server with mem_fraction=$MEM ==="
  python -m sglang.launch_server \
    --model Qwen/Qwen3.5-27B \
    --tp 1 \
    --port 25000 \
    --disable-cuda-graph \
    --mem-fraction-static $MEM &
  SERVER_PID=$!

  for i in $(seq 1 60); do
    if curl -s http://localhost:25000/health > /dev/null 2>&1; then
      echo "Server ready"
      break
    fi
    sleep 5
  done

  cd ~/sahas
  for PANELS in 1 2 3 5 7 10; do
    echo "=== mem=$MEM panels=$PANELS ==="
    python -m testbench efficiency \
      --num-panels $PANELS \
      --num-agents 5 \
      --mem-fraction $MEM
  done

  kill $SERVER_PID
  wait $SERVER_PID 2>/dev/null
  sleep 5
done

# Print summary
cd ~/sahas
python -m testbench efficiency --summarize

echo "=== Done ==="
