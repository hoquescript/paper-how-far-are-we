#!/bin/bash
#SBATCH --job-name=embedding
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --gpus=h100:1
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=/dev/null

set -euo pipefail

# Setup environment variables
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Get root directory
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"

# Setup logging
DATE=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="logs/$DATE"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/run.out"
exec 2>&1

# Load Python
module load python/3.14
# Create virtual environment
virtualenv --clear "$SLURM_TMPDIR/ENV"
source "$SLURM_TMPDIR/ENV/bin/activate"
# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index --no-cache \
  tree-sitter~=0.25.2 \
  tree-sitter-cpp~=0.23.0 \
  tree-sitter-python~=0.25.0 \
  tree_sitter-java~=0.23.0 \
  tree-sitter-typescript~=0.23.2 \
  tree-sitter-javascript~=0.25.0 \
  numpy pandas torch transformers scikit-learn joblib


TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

declare -a LANGUAGE_NAMES=("Javascript")
declare -a DATASETS=("javascript.csv")

# Exclude tasks that are not supported
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#DATASETS[@]}" ]; then
  echo "Unsupported SLURM_ARRAY_TASK_ID=$TASK_ID. Expected 0-$(( ${#DATASETS[@]} - 1 ))" >&2
  exit 1
fi

# Assign the data CSV file if the environment variable is not set
if [ -z "${DATA_CSV:-}" ]; then
  export DATA_CSV="$ROOT_DIR/data/aidev/${DATASETS[$TASK_ID]}"
fi

# Check if the data CSV file exists in that file path
if [ ! -f "$DATA_CSV" ]; then
  echo "DATA_CSV not found: $DATA_CSV" >&2
  exit 1
fi

echo "Running ${LANGUAGE_NAMES[$TASK_ID]} job with $DATA_CSV"

python main.py
