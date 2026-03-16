#!/bin/bash
#SBATCH --job-name=ast_svm
#SBATCH --array=0-1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p "$ROOT_DIR/logs"
cd "$ROOT_DIR"

module load python/3.14

virtualenv --clear "$SLURM_TMPDIR/ENV"
source "$SLURM_TMPDIR/ENV/bin/activate"

pip install --no-index --upgrade pip
pip install --no-index --no-cache \
    tree-sitter==0.25.2 \
    tree-sitter-cpp~=0.23.0 \
    tree-sitter-python~=0.25.0 \
    tree_sitter-java~=0.23.0 \
    numpy pandas torch transformers scikit-learn scipy sentencepiece

python -c "import tree_sitter, tree_sitter_cpp, tree_sitter_python, tree_sitter_java"
python -c "import torch; print(torch.__version__)"

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

if [ -z "${DATA_CSV:-}" ]; then
  if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
    export DATA_CSV="$ROOT_DIR/data/java.csv"
  else
    export DATA_CSV="$ROOT_DIR/data/python.csv"
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

python -m jobs.main
