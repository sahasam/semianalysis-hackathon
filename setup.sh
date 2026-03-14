#!/bin/bash
#SBATCH --job-name=setup-deps
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p ~/logs
source ~/miniforge3/bin/activate
conda activate bench
export HF_HOME=~/hf-cache

pip install langgraph langchain-openai langchain-core
