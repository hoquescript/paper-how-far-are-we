#!/bin/bash
#SBATCH --job-name=ast_svm
#SBATCH --array=0-1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

module load python scipy-stack
VENV=$HOME/venvs/how-far-are-we

source ~/venvs/how-far-are-we/bin/activate

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

mkdir -p "$ROOT_DIR/logs"
cd "$ROOT_DIR"

$VENV/bin/python -c "import torch; print(torch.__version__)"

# Pick dataset based on array index
if [ -z "${DATA_CSV:-}" ]; then
  if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
    export DATA_CSV=data/java.csv
  else
    export DATA_CSV=data/python.csv
  fi
fi

if [ ! -f "$DATA_CSV" ]; then
  echo "DATA_CSV not found: $DATA_CSV" >&2
  exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
  echo "Running Java job with $DATA_CSV"
else
  echo "Running Python job with $DATA_CSV"
fi

$VENV/bin/python -m jobs.main
