#!/bin/bash
#SBATCH --job-name=codegptsensor
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --array=0-3
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A_%a.out

# Submit from this directory so that logs/ resolves:
#   cd model/contrastive && sbatch train.sh
#
# Data comes from prepare_contrastive_dataset.py at the repo root:
#   uv run python prepare_contrastive_dataset.py

set -euo pipefail

# Locate the repo root by walking up to the directory holding pyproject.toml.
# SLURM copies the batch script to a spool dir, so BASH_SOURCE is unreliable here.
find_repo_root() {
  local dir="$1"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/pyproject.toml" ]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if ! REPO_ROOT="$(find_repo_root "$SUBMIT_DIR")"; then
  echo "Could not find the repo root (no pyproject.toml above $SUBMIT_DIR)." >&2
  echo "Submit from inside the repository, e.g. cd model/contrastive && sbatch train.sh" >&2
  exit 1
fi

RUN_DIR="$REPO_ROOT/model/contrastive/model"
DATA_ROOT="$REPO_ROOT/data/aidev_contrastive"
mkdir -p "$SUBMIT_DIR/logs"

module load python/3.14

virtualenv --clear "$SLURM_TMPDIR/ENV"
source "$SLURM_TMPDIR/ENV/bin/activate"

pip install --no-index --upgrade pip
# tree-sitter: CC wheelhouse for python/3.14 ships 0.25.2+computecanada, not 0.23.x.
# Language wheels are resolved from the same wheelhouse without strict pins.
pip install --no-index --no-cache-dir \
  numpy \
  pandas \
  torch \
  "transformers==4.57.6" \
  scikit-learn \
  scipy \
  sentencepiece \
  tree-sitter==0.25.2 \
  tree-sitter-cpp~=0.23.0 \
  tree-sitter-python~=0.25.0 \
  tree_sitter-java~=0.23.0 \
  tree-sitter-javascript~=0.25.0 \
  tree-sitter-typescript~=0.23.2 \

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

python -c "import tree_sitter, tree_sitter_cpp, tree_sitter_python, tree_sitter_java, tree_sitter_javascript, tree_sitter_typescript"
python -c "import torch; print(f'torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"

LANGUAGES=(
  "java"
  "python"
  "javascript"
  "typescript"
)
read -r -a REPRESENTATIONS <<< "${REPRESENTATIONS:-code}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#LANGUAGES[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID" >&2
  exit 1
fi

LANGUAGE="${LANGUAGES[$TASK_ID]}"

MODEL_NAME="${MODEL_NAME:-microsoft/unixcoder-base-nine}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BLOCK_SIZE="${BLOCK_SIZE:-400}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SEED="${SEED:-99}"

TRAIN_DATA_FILE="$DATA_ROOT/$LANGUAGE/train.jsonl"
EVAL_DATA_FILE="$DATA_ROOT/$LANGUAGE/valid.jsonl"
TEST_DATA_FILE="$DATA_ROOT/$LANGUAGE/test.jsonl"

for path in "$TRAIN_DATA_FILE" "$EVAL_DATA_FILE" "$TEST_DATA_FILE"; do
  if [ ! -f "$path" ]; then
    echo "Dataset file not found: $path" >&2
    echo "Generate it first: uv run python prepare_contrastive_dataset.py" >&2
    exit 1
  fi
done

echo "Repo root:  $REPO_ROOT"
echo "Running language=$LANGUAGE"
echo "Train file: $TRAIN_DATA_FILE"
echo "Eval file:  $EVAL_DATA_FILE"
echo "Test file:  $TEST_DATA_FILE"

# run.py does `from model import Model` and `from utils...`, both resolved
# relative to its own directory.
cd "$RUN_DIR"

for REPRESENTATION in "${REPRESENTATIONS[@]}"; do
  OUTPUT_DIR="$RUN_DIR/models_output/${LANGUAGE}_${REPRESENTATION}"

  echo "Running representation=$REPRESENTATION"
  echo "Output dir: $OUTPUT_DIR"

  python "$RUN_DIR/run.py" \
    --do_train \
    --representation "$REPRESENTATION" \
    --model_name_or_path "$MODEL_NAME" \
    --train_data_file "$TRAIN_DATA_FILE" \
    --eval_data_file "$EVAL_DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --block_size "$BLOCK_SIZE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --seed "$SEED" \
    --contrast

  python "$RUN_DIR/run.py" \
    --do_test \
    --representation "$REPRESENTATION" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --test_data_file "$TEST_DATA_FILE" \
    --block_size "$BLOCK_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --seed "$SEED"
done
