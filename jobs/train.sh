#!/bin/bash
#SBATCH --job-name=ast_svm
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1          # request 1 GPU (H100 on Fir)
#SBATCH --cpus-per-task=32         # tokenizer + AST parsing benefit from CPU
#SBATCH --mem=128G                  # adjust if your CSV is large
#SBATCH --output=logs/%x-%j.out

module load python scipy-stack
source ~/venvs/how-far-are-we/bin/activate

export DATA_CSV=data/java.csv

# Hugging Face caches (keep in HOME or PROJECT, not SCRATCH)
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

python scripts/embeddings/main.py
