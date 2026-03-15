#!/bin/bash
#SBATCH --job-name=2nd-brain
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p ~/logs ~/sabareesh
source ~/miniforge3/bin/activate
conda activate bench
export HF_HOME=~/hf-cache
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
export SGLANG_DISABLE_CUDNN_CHECK=1

pip install zeus-ml 2>/dev/null | tail -1

cd ~/sabareesh

# ---------------------------------------------------------------------------
# Start SGLang server
# ---------------------------------------------------------------------------
echo "Starting SGLang server..."
python -m sglang.launch_server \
  --model Qwen/Qwen3.5-27B \
  --tp 1 \
  --port 25000 &
SERVER_PID=$!

echo "Waiting for server..."
for i in $(seq 1 90); do
  curl -s http://localhost:25000/health > /dev/null 2>&1 && break
  sleep 5
done

if ! curl -s http://localhost:25000/health > /dev/null 2>&1; then
  echo "ERROR: Server failed to start"
  kill $SERVER_PID 2>/dev/null
  exit 1
fi
echo "Server ready!"

# ---------------------------------------------------------------------------
# Run second brain benchmark: 1, 2, 3, 4 brains on 20 eval tasks
# ---------------------------------------------------------------------------
python arc_second_brain.py \
  --data-dir arc-agi-2/data \
  --split evaluation \
  --n-tasks 20 \
  --brains 1 2 3 4

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Done."
