#!/bin/bash
#SBATCH --job-name=sfno_corrupt
#SBATCH --time=02:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ============================================================================
# SFNO embedding — corruption-severity sweep (Protocol 2).
#
# Builds a clean ERA5 reference latent distribution, then measures MMD (and FID
# unless --mmd-only) against six corruptions across a severity ladder. Only ERA5
# is needed (no forecast). Corruptions are applied in SFNO's standardized space.
#
# Defaults match the real-vs-forecast run: smallest model (5c/15x30), full
# flatten (2250-dim), MMD only. See submit_sfno_mmd.sh for the data/weights
# staging prerequisites (same SFNO_REPO + ERA5 5-var file).
#
#   sbatch scripts/submit_sfno_distances.sh
#   # or e.g.:  POOLING=grid MMD_ONLY=0 N_SAMPLES=2000 sbatch scripts/submit_sfno_distances.sh
# ============================================================================

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields/FeatureMetric

export SFNO_REPO="${SFNO_REPO:-$HOME/SFNO-Embedding}"
ERA5_5VAR="${ERA5_5VAR:-/work/scratch/$USER/sfno_data/era5_5var_2020.nc}"
CHANNELS="${CHANNELS:-5}"
RES="${RES:-15}"
POOLING="${POOLING:-flatten}"
N_SAMPLES="${N_SAMPLES:-500}"
N_STEPS="${N_STEPS:-9}"
SEED="${SEED:-0}"
MMD_ONLY="${MMD_ONLY:-1}"          # 1 = pass --mmd-only (needed for flatten)

mkdir -p plots

missing=0
for f in "$SFNO_REPO/weights/model_${CHANNELS}c_${RES}.pth" \
         "$SFNO_REPO/weights/static_fields.pth" "$ERA5_5VAR"; do
    [ -e "$f" ] || { echo "MISSING: $f"; missing=1; }
done
if [ "$missing" -ne 0 ]; then
    echo ""
    echo "Prerequisites missing — stage the SFNO repo/weights and the ERA5 5-var"
    echo "file (see submit_sfno_mmd.sh), then re-submit."
    exit 1
fi

mmd_flag=""
[ "$MMD_ONLY" = "1" ] && mmd_flag="--mmd-only"

echo "Starting SFNO corruption sweep on node: $(hostname)"
nvidia-smi || true
echo "SFNO_REPO=$SFNO_REPO  ERA5=$ERA5_5VAR"
echo "config: ${CHANNELS}c ${RES}  pool=$POOLING  n=$N_SAMPLES  steps=$N_STEPS  mmd_only=$MMD_ONLY"

python eval/eval_sfno_distances.py \
    --era5-path "$ERA5_5VAR" \
    --channels "$CHANNELS" \
    --res "$RES" \
    --pooling "$POOLING" \
    $mmd_flag \
    --n-samples "$N_SAMPLES" \
    --n-severity-steps "$N_STEPS" \
    --batch-size 16 \
    --num-workers 2 \
    --seed "$SEED"

echo "SFNO corruption sweep finished!"
