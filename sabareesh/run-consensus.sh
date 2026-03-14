#!/bin/bash
#SBATCH --job-name=living-model
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
  --port 25000 \
  --disable-cuda-graph &
SERVER_PID=$!

echo "Waiting for server..."
for i in $(seq 1 60); do
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
# Run 1: Single panel (baseline utilization)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  RUN 1: Single panel — 7 agents, living model topic"
echo "================================================================"
python consensus.py \
  --topic "What is the best architecture for a living, non-stateless AI model? Should persistent memory live inside the weights (continual learning / online fine-tuning), outside the weights (RAG / knowledge graphs / external memory), or in a hybrid? How do you handle catastrophic forgetting vs. stale knowledge? What are the right trade-offs between plasticity and stability, and how do you make the whole system auditable and reversible? Be specific about implementation — what concrete components, data flows, and failure modes matter most?" \
  --num-agents 7 \
  --max-rounds 15 \
  --temperature 0.8 \
  --max-tokens 500

# ---------------------------------------------------------------------------
# Run 2: Multi-panel (3 concurrent panels to saturate GPU)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  RUN 2: Multi-panel — 3 concurrent panels x 7 agents"
echo "================================================================"
python consensus.py \
  --multi-panel \
  --num-agents 7 \
  --max-rounds 15 \
  --temperature 0.8 \
  --max-tokens 500

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Done."
