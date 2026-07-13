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
# SFNO (4-field) embedding — corruption-severity sweep (Protocol 2).
#
# Builds a clean ERA5 reference latent distribution, then measures MMD (and FID
# unless --mmd-only) against six corruptions across a severity ladder. Only ERA5
# is needed (no forecast). Corruptions are applied in SFNO's standardized space;
# they operate on the raw 4-var (4,121,240) field and the wind corruptions touch
# only U10/V10.
#
# Defaults: smallest model (4c/15x28), full flatten (4*15*28=1680-dim), MMD only.
# See submit_sfno_mmd.sh for the weights staging prerequisites (same SFNO_REPO +
# weights_4fields/). Uses the standard team07 ERA5 file by default.
#
#   sbatch scripts/submit_sfno_distances.sh
#   # or e.g.:  CHANNELS=8 POOLING=grid MMD_ONLY=0 N_SAMPLES=2000 sbatch scripts/submit_sfno_distances.sh
# ============================================================================

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields/FeatureMetric

export SFNO_REPO="${SFNO_REPO:-$HOME/SFNO-Embedding}"
ERA5="${ERA5:-/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc}"
CHANNELS="${CHANNELS:-4}"          # 4 / 8 / 16  (31x60 only has 8c)
RES="${RES:-15}"                   # 15 -> 15x28, 31 -> 31x60
POOLING="${POOLING:-flatten}"
N_SAMPLES="${N_SAMPLES:-500}"
N_STEPS="${N_STEPS:-9}"
SEED="${SEED:-0}"
MMD_ONLY="${MMD_ONLY:-1}"          # 1 = pass --mmd-only (needed for flatten)

case "$RES" in
    15) HW="15x28" ;;
    31) HW="31x60" ;;
    *) echo "Unsupported RES=$RES (use 15 or 31)"; exit 1 ;;
esac

mkdir -p plots

missing=0
for f in "$SFNO_REPO/weights_4fields/model_${CHANNELS}c_${HW}_4fields.pth" \
         "$SFNO_REPO/weights_4fields/static_fields.pth" \
         "$SFNO_REPO/weights_4fields/normalization_means_4fields.pt" \
         "$SFNO_REPO/weights_4fields/normalization_stds_4fields.pt" \
         "$ERA5"; do
    [ -e "$f" ] || { echo "MISSING: $f"; missing=1; }
done
if [ "$missing" -ne 0 ]; then
    echo ""
    echo "Prerequisites missing — stage the SFNO repo/4-field weights (see"
    echo "submit_sfno_mmd.sh), then re-submit."
    exit 1
fi

mmd_flag=""
[ "$MMD_ONLY" = "1" ] && mmd_flag="--mmd-only"

echo "Starting SFNO 4-field corruption sweep on node: $(hostname)"
nvidia-smi || true
echo "SFNO_REPO=$SFNO_REPO  ERA5=$ERA5"
echo "config: ${CHANNELS}c ${HW}  pool=$POOLING  n=$N_SAMPLES  steps=$N_STEPS  mmd_only=$MMD_ONLY"

python eval/eval_sfno_distances.py \
    --era5-path "$ERA5" \
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
