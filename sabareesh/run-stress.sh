#!/bin/bash
#SBATCH --job-name=stress-test
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
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
# Start SGLang server — CUDA graphs enabled
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
# Panel scaling sweep — push concurrency from 5 to 70
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  PANEL SCALING SWEEP"
echo "================================================================"

for P in 1 3 5 7 10 14; do
  echo ""
  echo "=== $P panel(s) x 5 agents = $((P * 5)) max concurrent ==="
  python stress_test.py \
    --num-panels $P \
    --num-agents 5 \
    --max-rounds 8 \
    --temperature 0.8 \
    --max-tokens 400
done

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "All done."
