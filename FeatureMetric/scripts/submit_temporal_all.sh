#!/bin/bash
#SBATCH --job-name=tmp_all
#SBATCH --time=72:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=logs/tmp_all_%j.out
#SBATCH --error=logs/tmp_all_%j.err

# Combined run of all four temporal-dynamics experiments in a single SLURM job
# (cluster restricts to one running job per user). Order:
#   1. exp1 — diff   (X_t − X_{t−24h})
#   2. exp2 — concat ([X_{t−24h}, X_t])
#   3. exp3 — phase  ([X_t, X_t − X_{t−24h}])     (recommended formulation)
#   4. baseline — none (fresh static retrain at matched LR/epochs, last so the
#                       experimental cells are saved first if walltime is hit)
#
# Each phase trains MAE → I-JEPA → runs eval_real_vs_forecast (max-pool).
# A phase failure does NOT abort the remaining phases — see the wrapper.

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

# Deliberately NOT set -e: we want one failure to not nuke the other phases.
# Still keep pipefail for visibility within a phase.
set -o pipefail
export PYTHONUNBUFFERED=1
export EXTRACT_FEATURES_POOLING=max

cd ~/atmospheric-fields
mkdir -p logs

# ── Shared config ───────────────────────────────────────────────────────────
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
DELTA_HOURS=24
DATE_TAG="may_13"

# Per-phase output dirs
DIR_EXP1="/work/scratch/${USER}/results/${DATE_TAG}_temporal_exp1_diff_d512_maxpool"
DIR_EXP2="/work/scratch/${USER}/results/${DATE_TAG}_temporal_exp2_concat_d512_maxpool"
DIR_EXP3="/work/scratch/${USER}/results/${DATE_TAG}_temporal_exp3_phase_d512_maxpool"
DIR_BASE="/work/scratch/${USER}/results/${DATE_TAG}_temporal_baseline_d512_maxpool"

mkdir -p "$DIR_EXP1" "$DIR_EXP2" "$DIR_EXP3" "$DIR_BASE"

echo "Starting combined temporal run on: $(hostname)"
echo "Date tag: $DATE_TAG"
echo "MAE lr=$MAE_LR  | I-JEPA peak=$JEPA_LR | epochs=$EPOCHS"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Track phase outcomes for a summary at the end.
declare -A PHASE_STATUS

# ── Phase runner ────────────────────────────────────────────────────────────
# Args: 1=label, 2=output_dir, 3=temporal-mode flags (string)
run_phase () {
    local label="$1"
    local out_dir="$2"
    local mode="$3"
    local extra_flags="$4"

    echo
    echo "############################################################"
    echo "# PHASE: $label   (mode=$mode)"
    echo "# Output dir: $out_dir"
    echo "# Started at: $(date -u +%FT%TZ)"
    echo "############################################################"

    # Training MAE
    echo -e "\n--- $label: Train MAE ---"
    python train/train_mae.py \
        --epochs $EPOCHS \
        --batch-size $BATCH \
        --num-workers $WORKERS \
        --lr $MAE_LR \
        --model-size $MODEL_SIZE \
        --embed-dim $EMBED_DIM \
        --temporal-mode "$mode" \
        $extra_flags \
        --output-dir "$out_dir" \
        --lazy
    local mae_rc=$?

    # Training I-JEPA
    echo -e "\n--- $label: Train I-JEPA ---"
    python train/train_ijepa.py \
        --epochs $EPOCHS \
        --batch-size $BATCH \
        --num-workers $WORKERS \
        --start-lr $JEPA_START_LR \
        --lr $JEPA_LR \
        --final-lr $JEPA_FINAL_LR \
        --model-size $MODEL_SIZE \
        --embed-dim $EMBED_DIM \
        --temporal-mode "$mode" \
        $extra_flags \
        --output-dir "$out_dir" \
        --lazy
    local jepa_rc=$?

    # Eval (always attempt, even if one training partially failed — eval will
    # raise its own clear error if a checkpoint is missing)
    echo -e "\n--- $label: Real vs Forecast eval ---"
    python eval/eval_real_vs_forecast.py \
        --model-size $MODEL_SIZE \
        --embed-dim $EMBED_DIM \
        --temporal-mode "$mode" \
        $extra_flags \
        --n-samples $N_SAMPLES \
        --batch-size 32 \
        --num-workers $WORKERS \
        --baseline-pool all-years \
        --baseline-n-per-half 1000 \
        --output-dir "$out_dir"
    local eval_rc=$?

    if [ $mae_rc -eq 0 ] && [ $jepa_rc -eq 0 ] && [ $eval_rc -eq 0 ]; then
        PHASE_STATUS[$label]="OK"
    else
        PHASE_STATUS[$label]="FAIL(mae=$mae_rc, jepa=$jepa_rc, eval=$eval_rc)"
    fi
    echo "# $label finished at $(date -u +%FT%TZ) — ${PHASE_STATUS[$label]}"
}

# ── Phase 1: Exp1 (diff) ────────────────────────────────────────────────────
run_phase "exp1_diff"   "$DIR_EXP1" "diff"   "--delta-hours $DELTA_HOURS"

# ── Phase 2: Exp2 (concat) ──────────────────────────────────────────────────
run_phase "exp2_concat" "$DIR_EXP2" "concat" "--delta-hours $DELTA_HOURS"

# ── Phase 3: Exp3 (phase) ───────────────────────────────────────────────────
run_phase "exp3_phase"  "$DIR_EXP3" "phase"  "--delta-hours $DELTA_HOURS"

# ── Phase 4: static baseline (last, so experiments are saved first) ────────
run_phase "baseline"    "$DIR_BASE" "none"   ""

# ── Summary ────────────────────────────────────────────────────────────────
echo
echo "############################################################"
echo "# All phases finished at $(date -u +%FT%TZ)"
echo "############################################################"
for label in exp1_diff exp2_concat exp3_phase baseline; do
    echo "  $label : ${PHASE_STATUS[$label]:-NOT_RUN}"
done

echo
echo "Pull results with:"
echo "  scp -r ddemler@student-cluster.inf.ethz.ch:$DIR_EXP1 ."
echo "  scp -r ddemler@student-cluster.inf.ethz.ch:$DIR_EXP2 ."
echo "  scp -r ddemler@student-cluster.inf.ethz.ch:$DIR_EXP3 ."
echo "  scp -r ddemler@student-cluster.inf.ethz.ch:$DIR_BASE ."
