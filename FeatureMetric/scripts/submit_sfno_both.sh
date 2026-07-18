#!/bin/bash
#SBATCH --job-name=sfno_all
#SBATCH --time=03:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ============================================================================
# SFNO (4-field) — FULL eval suite in one job, run sequentially. For each of
# two configs it runs BOTH the real-vs-forecast and the corruption sweep:
#
#   config A: smallest 4c/15x28, full flatten (1680-dim), MMD only
#   config B: larger   8c/31x60, max pooling (feature_dim=8), FID + MMD
#
#   1/4  real-vs-forecast  A   → plots/sfno_real_vs_forecast/  (ERA5 vs Pangu+GraphCast)
#   2/4  real-vs-forecast  B   → plots/sfno_real_vs_forecast/
#   3/4  corruption sweep  A   → plots/sfno_distances/         (ERA5 vs 6 corruptions)
#   4/4  corruption sweep  B   → plots/sfno_distances/
#
# Real-vs-forecast uses the team07 ERA5/Pangu/GraphCast files (baked into the
# eval defaults). The corruption sweep needs ERA5 only ($ERA5 below).
#
# Every step is attempted even if an earlier one fails; the job exits non-zero
# if any step failed. See submit_sfno_mmd.sh for weights staging prerequisites
# (needs $SFNO_REPO/weights_4fields/).
#
#   sbatch scripts/submit_sfno_both.sh
#   # override shared knobs:  N_SAMPLES=2000 CORRUPT_N=1000 SEED=1 sbatch scripts/submit_sfno_both.sh
# ============================================================================

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields/FeatureMetric

export SFNO_REPO="${SFNO_REPO:-$HOME/SFNO-Embedding}"
ERA5="${ERA5:-/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc}"
N_SAMPLES="${N_SAMPLES:-1000}"     # real-vs-forecast: samples per pool
N_BOOT="${N_BOOT:-200}"            # real-vs-forecast: resamples for error bars (0=off)
CORRUPT_N="${CORRUPT_N:-500}"      # corruption sweep: reference samples
CORRUPT_STEPS="${CORRUPT_STEPS:-9}"
SEED="${SEED:-0}"

mkdir -p plots

# ── Preflight: both checkpoints + shared static/norm files + ERA5 ────────────
W="$SFNO_REPO/weights_4fields"
missing=0
for f in "$W/model_4c_15x28_4fields.pth" \
         "$W/model_8c_31x60_4fields.pth" \
         "$W/static_fields.pth" \
         "$W/normalization_means_4fields.pt" \
         "$W/normalization_stds_4fields.pt" \
         "$ERA5"; do
    [ -e "$f" ] || { echo "MISSING: $f"; missing=1; }
done
if [ "$missing" -ne 0 ]; then
    echo ""
    echo "Prerequisites missing — stage \$SFNO_REPO/weights_4fields/ (see"
    echo "submit_sfno_mmd.sh) and check \$ERA5, then re-submit."
    exit 1
fi

echo "Starting SFNO 4-field full eval suite on node: $(hostname)"
nvidia-smi || true
echo "SFNO_REPO=$SFNO_REPO  ERA5=$ERA5"
echo "N_SAMPLES=$N_SAMPLES  N_BOOT=$N_BOOT  CORRUPT_N=$CORRUPT_N  CORRUPT_STEPS=$CORRUPT_STEPS  SEED=$SEED"

# Attempt every step regardless of individual failures; track exit status.
set +e
rc=0
run () {  # run "<label>" <command...>
    local label="$1"; shift
    echo ""
    echo "================ $label ================"
    "$@"
    local r=$?
    [ "$r" -ne 0 ] && { echo "$label FAILED (exit $r)"; rc=$r; }
}

# --- Real vs forecast (ERA5 vs Pangu + GraphCast) ---
run "Run 1/4: real-vs-forecast 4c/15x28 flatten MMD" \
    python eval/eval_sfno_real_vs_forecast.py \
        --channels 4 --res 15 --pooling flatten --mmd-only \
        --n-samples "$N_SAMPLES" --n-boot "$N_BOOT" \
        --batch-size 16 --num-workers 2 --seed "$SEED"

run "Run 2/4: real-vs-forecast 8c/31x60 max FID+MMD" \
    python eval/eval_sfno_real_vs_forecast.py \
        --channels 8 --res 31 --pooling max \
        --n-samples "$N_SAMPLES" --n-boot "$N_BOOT" \
        --batch-size 16 --num-workers 2 --seed "$SEED"

# --- Corruption sweep (ERA5 vs 6 corruptions across a severity ladder) ---
run "Run 3/4: corruption sweep 4c/15x28 flatten MMD" \
    python eval/eval_sfno_distances.py \
        --era5-path "$ERA5" \
        --channels 4 --res 15 --pooling flatten --mmd-only \
        --n-samples "$CORRUPT_N" --n-severity-steps "$CORRUPT_STEPS" \
        --batch-size 16 --num-workers 2 --seed "$SEED"

run "Run 4/4: corruption sweep 8c/31x60 max FID+MMD" \
    python eval/eval_sfno_distances.py \
        --era5-path "$ERA5" \
        --channels 8 --res 31 --pooling max \
        --n-samples "$CORRUPT_N" --n-severity-steps "$CORRUPT_STEPS" \
        --batch-size 16 --num-workers 2 --seed "$SEED"

echo ""
if [ "$rc" -eq 0 ]; then
    echo "All four SFNO runs finished!"
else
    echo "Done with errors (last failing exit=$rc). Check the per-run FAILED lines above."
fi
exit "$rc"
