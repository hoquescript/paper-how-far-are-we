#!/bin/bash
#SBATCH --job-name=ast_svm
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --array=0-3
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
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
  tree-sitter-python~=0.25.0 \
  tree_sitter-java~=0.23.0 \
  tree-sitter-typescript==0.23.2 \
  tree-sitter-javascript==0.25.0 \
  numpy pandas torch "transformers==4.57.6" scikit-learn scipy sentencepiece

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

export EMBED_BATCH_SIZE=128

python -c "import tree_sitter, tree_sitter_cpp, tree_sitter_python, tree_sitter_java, tree_sitter_typescript, tree_sitter_javascript"
python -c "import torch; print(f'torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

declare -a LANGUAGE_NAMES=("Java" "Python" "TypeScript" "JavaScript")
declare -a DATASETS=("java.csv" "python.csv" "typescript.csv" "javascript.csv")

if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#DATASETS[@]}" ]; then
  echo "Unsupported SLURM_ARRAY_TASK_ID=$TASK_ID. Expected 0-$(( ${#DATASETS[@]} - 1 ))" >&2
  exit 1
fi

if [ -z "${DATA_CSV:-}" ]; then
  export DATA_CSV="$ROOT_DIR/data/aidev/${DATASETS[$TASK_ID]}"
fi

if [ ! -f "$DATA_CSV" ]; then
  echo "DATA_CSV not found: $DATA_CSV" >&2
  exit 1
fi

echo "Running ${LANGUAGE_NAMES[$TASK_ID]} job with $DATA_CSV"

python -m jobs.main
