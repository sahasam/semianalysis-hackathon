#!/bin/bash
#SBATCH --job-name=find-cuda
#SBATCH --gpus=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out

find / -name "libcuda.so*" 2>/dev/null
find / -name "libnvidia-ml.so*" 2>/dev/null | head -5
echo "---"
ldconfig -p 2>/dev/null | grep cuda
echo "---"
ls -la /usr/lib/x86_64-linux-gnu/libcuda* 2>/dev/null
ls -la /usr/local/cuda/lib64/stubs/libcuda* 2>/dev/null
ls -la /usr/local/cuda*/lib64/libcuda* 2>/dev/null
ls -la /usr/local/cuda*/targets/x86_64-linux/lib/stubs/libcuda* 2>/dev/null
