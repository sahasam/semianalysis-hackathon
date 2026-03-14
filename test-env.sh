#!/bin/bash
#SBATCH --job-name=test-env
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p ~/logs
source ~/miniforge3/bin/activate
conda activate bench
export HF_HOME=~/hf-cache
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH

echo "=== GPU check ==="
nvidia-smi
python -c "import torch; print(f'{torch.cuda.device_count()} GPUs available')"

echo "=== Starting SGLang server ==="
python -m sglang.launch_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tp 1 \
  --port 25000 \
  --disable-cuda-graph \
  --attention-backend flashinfer \
  --sampling-backend flashinfer &
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

echo "=== Checking model name ==="
curl -s http://localhost:25000/v1/models | python -m json.tool

echo "=== Smoke test: curl ==="
curl -s http://localhost:25000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role":"user","content":"Say hello in one sentence."}],
    "max_tokens": 30
  }' | python -m json.tool

echo "=== Smoke test: AIPerf (5 requests) ==="
aiperf profile \
  --model Qwen/Qwen2.5-7B-Instruct \
  --url http://localhost:25000 \
  --endpoint-type chat \
  --concurrency 1 \
  --request-count 5 \
  --streaming \
  --ui-type none

echo "=== All tests passed ==="
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
