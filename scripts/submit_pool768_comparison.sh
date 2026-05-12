#!/bin/bash
#SBATCH --job-name=pool768_cmp
#SBATCH --time=48:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=36G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# 768-dim MAE pooling comparison:
#   Test 1: embed_dim=384 + concat pooling (mean+max) → 768 effective latent dim
#   Test 2: embed_dim=768 + max pooling               → 768 latent dim

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields
mkdir -p results

echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

EPOCHS=150
BATCH=64
WORKERS=2
N_SAMPLES=500

DIR_CONCAT="results/pool768_concat384"
DIR_MAX="results/pool768_max768"
mkdir -p "$DIR_CONCAT" "$DIR_MAX"

# ── Test 1: embed_dim=384 + concat pooling (mean+max) → 768 ─────────────────
echo
echo "============================================================"
echo "Test 1: MAE embed_dim=384  +  concat pooling (768 effective)"
echo "============================================================"

echo "--- Training ---"
python train/train_mae.py \
    --epochs $EPOCHS \
    --batch-size $BATCH \
    --num-workers $WORKERS \
    --embed-dim 384 \
    --output-dir "$DIR_CONCAT" \
    --lazy

echo "--- Evaluating ---"
python eval/eval_real_vs_forecast.py \
    --mae-only \
    --pooling concat \
    --embed-dim 384 \
    --n-samples $N_SAMPLES \
    --batch-size 64 \
    --num-workers $WORKERS \
    --output-dir "$DIR_CONCAT"

# ── Test 2: embed_dim=768 + max pooling → 768 ───────────────────────────────
echo
echo "============================================================"
echo "Test 2: MAE embed_dim=768  +  max pooling"
echo "============================================================"

echo "--- Training ---"
python train/train_mae.py \
    --epochs $EPOCHS \
    --batch-size $BATCH \
    --num-workers $WORKERS \
    --embed-dim 768 \
    --output-dir "$DIR_MAX" \
    --lazy

echo "--- Evaluating ---"
python eval/eval_real_vs_forecast.py \
    --mae-only \
    --pooling max \
    --embed-dim 768 \
    --n-samples $N_SAMPLES \
    --batch-size 64 \
    --num-workers $WORKERS \
    --output-dir "$DIR_MAX"

echo
echo "============================================================"
echo "Done. Results in:"
echo "  $DIR_CONCAT/plots/real_vs_forecast/"
echo "  $DIR_MAX/plots/real_vs_forecast/"
echo "============================================================"
