#!/bin/bash
#SBATCH --job-name=sfno_rvf_both
#SBATCH --time=02:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ============================================================================
# SFNO (4-field) real-vs-forecast — TWO configs in one job, run sequentially:
#   1) smallest model 4c/15x28, full flatten (1680-dim), MMD only
#   2) larger model   8c/31x60, max pooling (feature_dim=8), FID + MMD
#
# Both compare ERA5 vs Pangu AND GraphCast (+ ERA5-self baseline) and write
# distinct plots in plots/sfno_real_vs_forecast/. Uses the team07 cluster
# ERA5/Pangu/GraphCast files (baked into the eval defaults — no --local).
#
# Both runs are attempted even if the first fails; the job exits non-zero if
# either failed. See submit_sfno_mmd.sh for weights staging prerequisites
# (needs $SFNO_REPO/weights_4fields/).
#
#   sbatch scripts/submit_sfno_both.sh
#   # override shared knobs:  N_SAMPLES=2000 SEED=1 sbatch scripts/submit_sfno_both.sh
# ============================================================================

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields/FeatureMetric

export SFNO_REPO="${SFNO_REPO:-$HOME/SFNO-Embedding}"
N_SAMPLES="${N_SAMPLES:-1000}"
SEED="${SEED:-0}"

mkdir -p plots

# ── Preflight: both checkpoints + shared static/norm files must be present ───
W="$SFNO_REPO/weights_4fields"
missing=0
for f in "$W/model_4c_15x28_4fields.pth" \
         "$W/model_8c_31x60_4fields.pth" \
         "$W/static_fields.pth" \
         "$W/normalization_means_4fields.pt" \
         "$W/normalization_stds_4fields.pt"; do
    [ -e "$f" ] || { echo "MISSING: $f"; missing=1; }
done
if [ "$missing" -ne 0 ]; then
    echo ""
    echo "SFNO 4-field weights missing — stage \$SFNO_REPO/weights_4fields/ (see"
    echo "submit_sfno_mmd.sh), then re-submit."
    exit 1
fi

echo "Starting SFNO 4-field real-vs-forecast (two configs) on node: $(hostname)"
nvidia-smi || true
echo "SFNO_REPO=$SFNO_REPO  N_SAMPLES=$N_SAMPLES  SEED=$SEED"

# Attempt both runs regardless of individual failures; track exit status.
set +e
rc=0

echo ""
echo "================ Run 1/2: 4c/15x28, flatten, MMD only ================"
python eval/eval_sfno_real_vs_forecast.py \
    --channels 4 --res 15 --pooling flatten --mmd-only \
    --n-samples "$N_SAMPLES" --batch-size 16 --num-workers 2 --seed "$SEED"
r1=$?; [ "$r1" -ne 0 ] && { echo "Run 1 FAILED (exit $r1)"; rc=$r1; }

echo ""
echo "================ Run 2/2: 8c/31x60, max pooling, FID + MMD ============"
python eval/eval_sfno_real_vs_forecast.py \
    --channels 8 --res 31 --pooling max \
    --n-samples "$N_SAMPLES" --batch-size 16 --num-workers 2 --seed "$SEED"
r2=$?; [ "$r2" -ne 0 ] && { echo "Run 2 FAILED (exit $r2)"; rc=$r2; }

echo ""
if [ "$rc" -eq 0 ]; then
    echo "Both SFNO real-vs-forecast runs finished!"
else
    echo "Done with errors (run1=$r1, run2=$r2)."
fi
exit "$rc"
