#!/bin/bash
#SBATCH --job-name=ast_svm
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
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
  numpy pandas torch "transformers==4.57.6" scikit-learn scipy sentencepiece

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

export EMBED_BATCH_SIZE=128

python -c "import tree_sitter, tree_sitter_cpp, tree_sitter_python, tree_sitter_java"
python -c "import torch; print(f'torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"

if [ -z "${DATA_CSV:-}" ]; then
  if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
    export DATA_CSV="$ROOT_DIR/data/aidev/java.csv"
  else
    export DATA_CSV="$ROOT_DIR/data/aidev/python.csv"
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
