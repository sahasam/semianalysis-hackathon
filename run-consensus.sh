#!/bin/bash
#SBATCH --job-name=consensus
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p ~/logs
source ~/miniforge3/bin/activate
conda activate bench
export HF_HOME=~/hf-cache
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
export SGLANG_DISABLE_CUDNN_CHECK=1

# ---------------------------------------------------------------------------
# Start SGLang server
# ---------------------------------------------------------------------------
echo "Starting SGLang server..."
python -m sglang.launch_server \
  --model Qwen/Qwen3.5-27B \
  --tp 1 \
  --port 25000 \
  --disable-cuda-graph &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 60); do
  curl -s http://localhost:25000/health > /dev/null 2>&1 && break
  sleep 5
done

# Verify server is up
if ! curl -s http://localhost:25000/health > /dev/null 2>&1; then
  echo "ERROR: Server failed to start"
  kill $SERVER_PID 2>/dev/null
  exit 1
fi
echo "Server ready!"

# ---------------------------------------------------------------------------
# Run consensus
# ---------------------------------------------------------------------------
python consensus.py \
  --topic "What is the most important challenge in AI safety and how should it be addressed?" \
  --num-agents 5 \
  --max-rounds 5 \
  --temperature 0.7 \
  --max-tokens 300

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Done."
