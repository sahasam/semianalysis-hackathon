#!/bin/bash
#SBATCH --job-name=detail-prof
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
# Run detailed profiling — serial then parallel, same topic
# ---------------------------------------------------------------------------
python detailed_profile.py \
  --topic "What is the best architecture for a living, non-stateless AI model? Should persistent memory live inside the weights, outside, or in a hybrid? How do you handle catastrophic forgetting vs. stale knowledge?" \
  --num-agents 5 \
  --max-rounds 5 \
  --temperature 0.7 \
  --max-tokens 400

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Done."
