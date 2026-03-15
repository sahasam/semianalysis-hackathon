#!/bin/bash
#SBATCH --job-name=energy-exp
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
 
set -e
mkdir -p ~/logs
source ~/miniforge3/bin/activate
conda activate bench
export HF_HOME=~/hf-cache
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
export TMPDIR=/tmp/$USER-exp
export PIP_CACHE_DIR=$TMPDIR/pip-cache
mkdir -p $TMPDIR $PIP_CACHE_DIR
 
# === Install extra deps (skips quickly if already installed) ===
echo "=== Installing zeus-ml, langgraph, langchain-openai ==="
pip install zeus-ml langgraph langchain-openai
 
# === Start SGLang server ===
echo "=== Starting SGLang server on port 25000 ==="
python -m sglang.launch_server \
  --model Qwen/Qwen2.5-7B-Instruct \
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
 
# === Verify model name ===
echo "=== Model served ==="
curl -s http://localhost:25000/v1/models | python -m json.tool
 
# === Run experiment ===
echo "=== Running experiment.py ==="
python ~/experiment.py
 
# === Cleanup ===
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
rm -rf $TMPDIR
echo "=== Done ==="
