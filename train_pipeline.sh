#!/bin/bash
#SBATCH --job-name=srcml-pipeline
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --gpus=h100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/dev/null

set -euo pipefail

# Setup environment variables
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Get root directory
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"

# Archive previous logs
mv -f "$ROOT_DIR/logs"/* "$ROOT_DIR/logs_history/" 2>/dev/null || true

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

# Assign the data CSV file if the environment variable is not set
if [ -z "${DATA_CSV:-}" ]; then
  export DATA_CSV="$ROOT_DIR/data/hmcorp_xml.csv"
fi

# Check if the data CSV file exists
if [ ! -f "$DATA_CSV" ]; then
  echo "DATA_CSV not found: $DATA_CSV" >&2
  exit 1
fi

echo "Running Java (HMCorp Pipeline) job with $DATA_CSV"

OUTPUT_DIR="$ROOT_DIR/embeddings"

# Pass --sample <N> to do a quick run on a subset; omit for full training
# SAMPLE=200 sbatch train_pipeline.sh
SAMPLE_ARG=""
if [ -n "${SAMPLE:-}" ]; then
  SAMPLE_ARG="--sample $SAMPLE"
fi

python -m model.metrics.srcml.pipeline \
  --output-dir "$OUTPUT_DIR" \
  $SAMPLE_ARG
