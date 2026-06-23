#!/bin/bash
#SBATCH --job-name=sfno_rvf
#SBATCH --time=02:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ============================================================================
# SFNO (4-field) embedding — real (ERA5) vs 24h forecast (Pangu + GraphCast).
#
# Produces FID/MMD bars + joint-PCA scatter, the same plots as the MAE/I-JEPA
# eval_real_vs_forecast.py. The 4-field checkpoints drop precipitation, so SFNO
# reads the SAME standard 4-var ERA5/Pangu/GraphCast files MAE/I-JEPA use — no
# special 5-var download, and Pangu is no longer excluded.
#
# Default config: 8c/31x60 (best RMSE/SSIM of the 4-field set), mean pooling
# (feature_dim=8 → FID is valid). Override via env (CHANNELS/RES/POOLING/...).
#
# ── PREREQUISITES (stage on the LOGIN node before sbatch) ────────────────────
# SFNO repo + 4-field weights at $SFNO_REPO (default ~/SFNO-Embedding):
#     git clone <sfno-embedding-url> ~/SFNO-Embedding
#     # copy the 4-field weights into ~/SFNO-Embedding/weights_4fields/ :
#     #   model_<C>c_<HxW>_4fields.pth  (e.g. model_8c_31x60_4fields.pth)
#     #   static_fields.pth
#     #   normalization_means_4fields.pt  normalization_stds_4fields.pt
# Data: the standard team07 ERA5/Pangu/GraphCast files (cluster defaults below)
# are read directly — nothing to download.
#
# Then:  sbatch scripts/submit_sfno_mmd.sh
#   # or e.g.:  CHANNELS=16 RES=15 POOLING=grid sbatch scripts/submit_sfno_mmd.sh
#   #           POOLING=flatten MMD_ONLY=1 sbatch scripts/submit_sfno_mmd.sh
# ============================================================================

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields/FeatureMetric

# ── Config (override via env) ───────────────────────────────────────────────
export SFNO_REPO="${SFNO_REPO:-$HOME/SFNO-Embedding}"
CHANNELS="${CHANNELS:-8}"          # 4 / 8 / 16  (31x60 only has 8c)
RES="${RES:-31}"                   # 15 -> 15x28, 31 -> 31x60
POOLING="${POOLING:-mean}"         # mean/max/meanstd/grid/flatten
N_SAMPLES="${N_SAMPLES:-1000}"
SEED="${SEED:-0}"
MMD_ONLY="${MMD_ONLY:-0}"          # 1 = pass --mmd-only (needed for flatten)

# Resolve HxW for the weight-file preflight check.
case "$RES" in
    15) HW="15x28" ;;
    31) HW="31x60" ;;
    *) echo "Unsupported RES=$RES (use 15 or 31)"; exit 1 ;;
esac

mkdir -p plots

# ── Preflight: fail fast with clear guidance if prerequisites are missing ────
missing=0
for f in "$SFNO_REPO/weights_4fields/model_${CHANNELS}c_${HW}_4fields.pth" \
         "$SFNO_REPO/weights_4fields/static_fields.pth" \
         "$SFNO_REPO/weights_4fields/normalization_means_4fields.pt" \
         "$SFNO_REPO/weights_4fields/normalization_stds_4fields.pt"; do
    if [ ! -e "$f" ]; then
        echo "MISSING: $f"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo ""
    echo "SFNO 4-field weights missing — see the PREREQUISITES block in this script."
    exit 1
fi

mmd_flag=""
[ "$MMD_ONLY" = "1" ] && mmd_flag="--mmd-only"

echo "Starting SFNO 4-field real-vs-forecast eval on node: $(hostname)"
nvidia-smi || true
echo "SFNO_REPO=$SFNO_REPO"
echo "config: ${CHANNELS}c ${HW}  pool=$POOLING  n=$N_SAMPLES  mmd_only=$MMD_ONLY"

# No --local: uses the team07 cluster ERA5/Pangu/GraphCast defaults baked into
# the eval script. Override with --era5-path/--pangu-path/--graphcast-path if needed.
python eval/eval_sfno_real_vs_forecast.py \
    --channels "$CHANNELS" \
    --res "$RES" \
    --pooling "$POOLING" \
    $mmd_flag \
    --n-samples "$N_SAMPLES" \
    --batch-size 16 \
    --num-workers 2 \
    --seed "$SEED"

echo "SFNO real-vs-forecast eval finished!"
