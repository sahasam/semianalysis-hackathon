#!/bin/bash
#SBATCH --job-name=efficiency
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
# Sweep: KV cache fraction x concurrency
#
# mem_fraction_static controls how much VRAM goes to KV cache vs weights
# Default ~0.79 — we try 0.80, 0.88, 0.93
# At each cache size, sweep panels to find peak J/token efficiency
# ---------------------------------------------------------------------------

for MEM_FRAC in 0.80 0.88 0.93; do
  echo ""
  echo "################################################################"
  echo "  KV CACHE FRACTION: $MEM_FRAC"
  echo "################################################################"

  # Start server with this cache config
  echo "Starting SGLang (mem_fraction_static=$MEM_FRAC)..."
  python -m sglang.launch_server \
    --model Qwen/Qwen3.5-27B \
    --tp 1 \
    --port 25000 \
    --mem-fraction-static $MEM_FRAC &
  SERVER_PID=$!

  for i in $(seq 1 90); do
    curl -s http://localhost:25000/health > /dev/null 2>&1 && break
    sleep 5
  done

  if ! curl -s http://localhost:25000/health > /dev/null 2>&1; then
    echo "ERROR: Server failed to start with mem_fraction=$MEM_FRAC"
    kill $SERVER_PID 2>/dev/null
    continue
  fi

  echo "Server ready (mem_fraction=$MEM_FRAC)"

  # Sweep concurrency at this cache size
  for P in 1 3 7 14 20 30; do
    echo ""
    echo "--- mem=$MEM_FRAC panels=$P concurrent=$((P * 5)) ---"
    python efficiency_sweep.py \
      --num-panels $P \
      --num-agents 5 \
      --max-rounds 6 \
      --temperature 0.8 \
      --max-tokens 400 \
      --mem-fraction $MEM_FRAC
  done

  # Kill server before next config
  kill $SERVER_PID
  wait $SERVER_PID 2>/dev/null
  sleep 5
done

echo ""
echo "================================================================"
echo "  ALL SWEEPS COMPLETE — generating summary"
echo "================================================================"
python efficiency_sweep.py --summarize

echo "Done."
