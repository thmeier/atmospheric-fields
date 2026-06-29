#!/bin/bash
#SBATCH --job-name=realism
#SBATCH --time=24:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=logs/june29_slurm_%j.out
#SBATCH --error=logs/june29_slurm_%j.err

# ============================================================================
# Contrastive realism metric — fully self-contained pipeline in one job.
# Everything lands under a single dated output dir ($OUTDIR), so after the job
# finishes you can scp that one folder locally and have all artifacts:
#
#   1. Train         RealismMetric from scratch on ERA5 + subtle corruptions
#                    → $OUTDIR/best_realism_model_<size>.pth, data_mean/std.npy
#   2. PSD diagnostic ERA5↔forecast high-wavenumber deficit (classical baseline)
#                    → $OUTDIR/plots/psd/june29_psd_era5_vs_forecast.png
#   3. Paired eval   valid-time-paired ERA5 vs Pangu/GraphCast, WITH bootstrap
#                    uncertainties (CIs on Δr / AUC + mean-score bar charts)
#                    → $OUTDIR/plots/realism_forecast/june29_*.png
#
# The slurm .out/.err are copied into $OUTDIR at the end too, so a single
# `scp -r ddemler@...:~/atmospheric-fields/FeatureMetric/runs/june29 .` grabs all.
#
#   sbatch scripts/submit_realism.sh
#   # override knobs: EPOCHS=150 BATCH=64 N_EVAL=1000 sbatch scripts/submit_realism.sh
# ============================================================================

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields/FeatureMetric
mkdir -p logs

RUN_TAG="${RUN_TAG:-june29}"
OUTDIR="${OUTDIR:-runs/$RUN_TAG}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-64}"
LR="${LR:-1e-3}"
MODEL_SIZE="${MODEL_SIZE:-twin}"
N_EVAL="${N_EVAL:-1000}"
N_BOOT="${N_BOOT:-200}"
SEED="${SEED:-0}"
mkdir -p "$OUTDIR"

echo "Realism pipeline on node: $(hostname)"
nvidia-smi || true
echo "RUN_TAG=$RUN_TAG OUTDIR=$OUTDIR EPOCHS=$EPOCHS BATCH=$BATCH LR=$LR"
echo "MODEL_SIZE=$MODEL_SIZE N_EVAL=$N_EVAL N_BOOT=$N_BOOT SEED=$SEED"

# Train first: it writes data_mean/std.npy into $OUTDIR, which the PSD step needs.
echo ""
echo "================ 1/3: Train realism metric ================"
python train/train_realism.py \
    --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
    --model-size "$MODEL_SIZE" --output-dir "$OUTDIR"

echo ""
echo "================ 2/3: PSD diagnostic (baseline) ================"
python eval/psd_diagnostic.py \
    --n-samples 500 --seed "$SEED" \
    --output-dir "$OUTDIR" --run-tag "$RUN_TAG"

echo ""
echo "================ 3/3: Paired forecast eval (with uncertainties) ================"
python eval/eval_realism_forecast.py \
    --n-samples "$N_EVAL" --model-size "$MODEL_SIZE" \
    --batch-size 32 --n-boot "$N_BOOT" --seed "$SEED" \
    --output-dir "$OUTDIR" --run-tag "$RUN_TAG"

# Copy slurm logs into the output dir so one scp grabs everything.
cp -f "logs/${RUN_TAG}_slurm_${SLURM_JOB_ID}.out" "$OUTDIR/" 2>/dev/null || true
cp -f "logs/${RUN_TAG}_slurm_${SLURM_JOB_ID}.err" "$OUTDIR/" 2>/dev/null || true

echo ""
echo "Realism pipeline finished. All artifacts in: $OUTDIR"
echo "  scp -r ddemler@student-cluster.inf.ethz.ch:~/atmospheric-fields/FeatureMetric/$OUTDIR ."
