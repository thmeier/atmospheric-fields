#!/bin/bash
#SBATCH --job-name=tmp_exp3_phase
#SBATCH --time=24:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=logs/tmp_exp3_phase_%j.out
#SBATCH --error=logs/tmp_exp3_phase_%j.err

# Experiment 3: phase-space concatenation X_phase = [X_t, X_t − X_{t−24h}] (8ch)
# Strongest formulation: provides both "position" (absolute state, abs-norm)
# and "velocity" (temporal derivative, diff-norm) channels. Diff half is clipped
# to ±15σ in compose_temporal_input to handle forecast hallucinations safely.

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

set -eo pipefail
export PYTHONUNBUFFERED=1
export EXTRACT_FEATURES_POOLING=max

cd ~/atmospheric-fields
mkdir -p logs

OUTPUT_DIR="/work/scratch/${USER}/results/may_13_temporal_exp3_phase_d512_maxpool"
MODEL_SIZE="twin"
EMBED_DIM=512
EPOCHS=150
BATCH=64
WORKERS=2
MAE_LR=1.5e-3
JEPA_START_LR=3e-4
JEPA_LR=7.5e-4
JEPA_FINAL_LR=1.5e-6
N_SAMPLES=500
TEMPORAL_MODE="phase"
DELTA_HOURS=24

mkdir -p "$OUTPUT_DIR"

echo "Starting temporal EXP3 (mode=$TEMPORAL_MODE Δt=${DELTA_HOURS}h) on: $(hostname)"
echo "Output: $OUTPUT_DIR"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# ── Phase 1: Train MAE ───────────────────────────────────────────────────────
echo -e "\n============================================================"
echo "Phase 1: Train MAE  (temporal-mode=$TEMPORAL_MODE)"
echo "============================================================"
python train/train_mae.py \
    --epochs $EPOCHS \
    --batch-size $BATCH \
    --num-workers $WORKERS \
    --lr $MAE_LR \
    --model-size $MODEL_SIZE \
    --embed-dim $EMBED_DIM \
    --temporal-mode $TEMPORAL_MODE \
    --delta-hours $DELTA_HOURS \
    --output-dir "$OUTPUT_DIR" \
    --lazy

# ── Phase 2: Train I-JEPA ────────────────────────────────────────────────────
echo -e "\n============================================================"
echo "Phase 2: Train I-JEPA  (temporal-mode=$TEMPORAL_MODE)"
echo "============================================================"
python train/train_ijepa.py \
    --epochs $EPOCHS \
    --batch-size $BATCH \
    --num-workers $WORKERS \
    --start-lr $JEPA_START_LR \
    --lr $JEPA_LR \
    --final-lr $JEPA_FINAL_LR \
    --model-size $MODEL_SIZE \
    --embed-dim $EMBED_DIM \
    --temporal-mode $TEMPORAL_MODE \
    --delta-hours $DELTA_HOURS \
    --output-dir "$OUTPUT_DIR" \
    --lazy

# ── Phase 3: Real vs Forecast eval ───────────────────────────────────────────
echo -e "\n============================================================"
echo "Phase 3: Real vs Forecast — ERA5 vs Pangu / GraphCast (max pool, $TEMPORAL_MODE)"
echo "============================================================"
python eval/eval_real_vs_forecast.py \
    --model-size $MODEL_SIZE \
    --embed-dim $EMBED_DIM \
    --temporal-mode $TEMPORAL_MODE \
    --delta-hours $DELTA_HOURS \
    --n-samples $N_SAMPLES \
    --batch-size 32 \
    --num-workers $WORKERS \
    --output-dir "$OUTPUT_DIR"

echo -e "\nDone. Results: $OUTPUT_DIR/plots/real_vs_forecast/"
echo "Pull with: scp -r ddemler@student-cluster.inf.ethz.ch:$OUTPUT_DIR ."
