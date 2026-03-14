#!/bin/bash
#SBATCH --job-name=bench-sweep
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ---------------------------------------------------------------------------
# Sweep SGLang server configs and consensus parameters to find the sweet spot
# for communication-heavy multi-agent workloads.
#
# What we're testing:
#   1. Baseline AIPerf benchmark at different concurrency levels
#   2. Consensus runs varying: agent count, max_tokens, temperature
#   3. Compare wall time, latency, effective req/s
# ---------------------------------------------------------------------------

mkdir -p ~/logs
source ~/miniforge3/bin/activate
conda activate bench
export HF_HOME=~/hf-cache
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
export SGLANG_DISABLE_CUDNN_CHECK=1

RESULTS_DIR="results/sweep-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

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
# Phase 1: AIPerf baseline — raw throughput at different concurrency levels
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  PHASE 1: AIPerf baseline benchmarks"
echo "================================================================"

for C in 1 5 10 20 50; do
  echo ""
  echo "--- AIPerf concurrency=$C ---"
  aiperf profile \
    --model Qwen/Qwen3.5-27B \
    --url http://localhost:25000 \
    --endpoint-type chat \
    --concurrency $C \
    --request-count 100 \
    --streaming \
    --ui-type none \
    2>&1 | tee "$RESULTS_DIR/aiperf-c${C}.txt"
done

# ---------------------------------------------------------------------------
# Phase 2: Consensus sweep — vary agent count
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  PHASE 2: Consensus — varying agent count"
echo "================================================================"

TOPIC="What is the most important challenge in AI safety and how should it be addressed?"

for N in 2 3 5 7; do
  echo ""
  echo "--- Consensus: $N agents ---"
  python consensus.py \
    --topic "$TOPIC" \
    --num-agents $N \
    --max-rounds 3 \
    --temperature 0.7 \
    --max-tokens 300 \
    2>&1 | tee "$RESULTS_DIR/consensus-agents${N}.txt"
done

# ---------------------------------------------------------------------------
# Phase 3: Consensus sweep — vary max_tokens (response length)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  PHASE 3: Consensus — varying response length"
echo "================================================================"

for T in 100 200 300 500; do
  echo ""
  echo "--- Consensus: max_tokens=$T ---"
  python consensus.py \
    --topic "$TOPIC" \
    --num-agents 5 \
    --max-rounds 3 \
    --temperature 0.7 \
    --max-tokens $T \
    2>&1 | tee "$RESULTS_DIR/consensus-tokens${T}.txt"
done

# ---------------------------------------------------------------------------
# Phase 4: Consensus sweep — vary temperature (convergence speed)
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  PHASE 4: Consensus — varying temperature"
echo "================================================================"

for TEMP in 0.3 0.5 0.7 1.0; do
  TEMP_LABEL=$(echo $TEMP | tr '.' '_')
  echo ""
  echo "--- Consensus: temperature=$TEMP ---"
  python consensus.py \
    --topic "$TOPIC" \
    --num-agents 5 \
    --max-rounds 5 \
    --temperature $TEMP \
    --max-tokens 300 \
    2>&1 | tee "$RESULTS_DIR/consensus-temp${TEMP_LABEL}.txt"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  SWEEP COMPLETE"
echo "================================================================"
echo "Results saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Done."
