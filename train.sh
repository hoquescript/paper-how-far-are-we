#!/bin/bash
#SBATCH --job-name=ast_svm
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out

module load python scipy-stack
source ~/venvs/how-far-are-we/bin/activate

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# Pick dataset based on array index
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  export DATA_CSV=data/java.csv
else
  export DATA_CSV=data/python.csv
fi

python scripts/embeddings/main.py
