#!/bin/bash
#SBATCH --job-name=atm_512_may07
#SBATCH --time=48:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=logs/512_may07_%j.out
#SBATCH --error=logs/512_may07_%j.err

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

set -eo pipefail
export PYTHONUNBUFFERED=1

# ── Run from repo root regardless of where sbatch was invoked ────────────────
cd ~/atmospheric-fields
mkdir -p logs

# ── Configuration ────────────────────────────────────────────────────────────
# NOTE: scratch is cleaned every 1–7 days; scp results off promptly after the run.
OUTPUT_DIR="/work/scratch/${USER}/results/may_07_512_encoder"
EMBED_DIM=512
MODEL_SIZE="twin"
mkdir -p "$OUTPUT_DIR"

echo "Starting 512-encoder pipeline on node: $(hostname)"
echo "Output directory: $OUTPUT_DIR"
echo "Embed dim: $EMBED_DIM, Model size: $MODEL_SIZE"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# ── Phase 1: Train MAE (twin, embed_dim=512) ─────────────────────────────────
echo -e "\n============================================================"
echo "Phase 1: Training MAE (twin, embed_dim=$EMBED_DIM)"
echo "============================================================"
python train/train_mae.py \
    --model-size "$MODEL_SIZE" \
    --embed-dim "$EMBED_DIM" \
    --epochs 100 \
    --batch-size 64 \
    --num-workers 2 \
    --stats-chunk-size 64 \
    --output-dir "$OUTPUT_DIR" \
    --lazy

# ── Phase 2: Train I-JEPA (twin, embed_dim=512) ──────────────────────────────
echo -e "\n============================================================"
echo "Phase 2: Training I-JEPA (twin, embed_dim=$EMBED_DIM)"
echo "============================================================"
python train/train_ijepa.py \
    --model-size "$MODEL_SIZE" \
    --embed-dim "$EMBED_DIM" \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 2 \
    --early-stopping-patience 15 \
    --output-dir "$OUTPUT_DIR" \
    --lazy

# ── Phase 3: Probe evaluation (corruption regression, both models) ───────────
echo -e "\n============================================================"
echo "Phase 3: Probe evaluation — corruption severity regression"
echo "============================================================"
python eval/eval_probe.py \
    --model both \
    --mae-size "$MODEL_SIZE" \
    --ijepa-size "$MODEL_SIZE" \
    --mae-embed-dim "$EMBED_DIM" \
    --ijepa-embed-dim "$EMBED_DIM" \
    --batch-size 64 \
    --num-workers 2 \
    --n-probe-samples 2000 \
    --output-dir "$OUTPUT_DIR" \
    --lazy

# ── Phase 4: Distance metrics (FID/MMD vs corruptions, both models) ──────────
echo -e "\n============================================================"
echo "Phase 4: Distance metrics — FID/MMD vs corruption severity"
echo "============================================================"
python eval/eval_distances.py \
    --model both \
    --mae-size "$MODEL_SIZE" \
    --ijepa-size "$MODEL_SIZE" \
    --mae-embed-dim "$EMBED_DIM" \
    --ijepa-embed-dim "$EMBED_DIM" \
    --batch-size 64 \
    --num-workers 2 \
    --n-severity-steps 15 \
    --output-dir "$OUTPUT_DIR" \
    --lazy

# ── Phase 5: Real vs Forecast (ERA5 vs Pangu / GraphCast, both models) ───────
echo -e "\n============================================================"
echo "Phase 5: Real vs Forecast — ERA5 vs Pangu / GraphCast"
echo "============================================================"
python eval/eval_real_vs_forecast.py \
    --model-size "$MODEL_SIZE" \
    --embed-dim "$EMBED_DIM" \
    --n-samples 500 \
    --batch-size 32 \
    --num-workers 2 \
    --output-dir "$OUTPUT_DIR"

echo -e "\n============================================================"
echo "Pipeline finished successfully."
echo "All artifacts under: $OUTPUT_DIR"
echo "  Checkpoints + stats: $OUTPUT_DIR/*.pth, *.npy"
echo "  Plots:               $OUTPUT_DIR/plots/"
echo "Pull with:"
echo "  scp -r ddemler@student-cluster.inf.ethz.ch:$OUTPUT_DIR ."
echo "============================================================"
