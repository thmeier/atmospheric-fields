#!/bin/bash
#SBATCH --job-name=sfno_mmd
#SBATCH --time=02:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ============================================================================
# SFNO embedding — real (ERA5) vs forecast (GraphCast) MMD, smallest model.
#
# Uses the 5c/15x30 SFNO checkpoint with FULL flatten (concatenate the whole
# (5,15,30)=2250-dim latent) and MMD only. MMD needs only pairwise distances,
# so high dim is fine; the smallest model keeps the vector at 2250 dims so the
# RBF median-bandwidth heuristic stays well-conditioned. FID is skipped (its
# covariance is singular at this dimension).
#
# ── PREREQUISITES (stage these on the LOGIN node before sbatch) ──────────────
# Compute nodes usually have no internet, so download data + weights first.
#
# 1) SFNO repo + weights at $SFNO_REPO (default ~/SFNO-Embedding):
#      git clone <sfno-embedding-url> ~/SFNO-Embedding
#      # download weights/ (~130 MB) per SFNO-Embedding/README.md into
#      # ~/SFNO-Embedding/weights/  (needs at least model_5c_15.pth,
#      # static_fields.pth, normalization_means.pt, normalization_stds.pt)
#
# 2) 5-var ERA5 + GraphCast NetCDF (with total_precipitation_6hr) in data/:
#      cd ~/atmospheric-fields/FeatureMetric
#      python scripts/download_forecast_netcdf.py era5 \
#          data/era5_5var_2020.nc -s 2020-01-01 -e 2020-12-31 --sfno-vars
#      python scripts/download_forecast_netcdf.py graphcast \
#          data/graphcast_5var_2020_lead24h.nc -s 2020-01-01 -e 2020-12-31 \
#          --lead-hours 24 --sfno-vars
#
# Then:  sbatch scripts/submit_sfno_mmd.sh
# Override paths/config via env vars, e.g.:
#   CHANNELS=5 RES=15 N_SAMPLES=1000 sbatch scripts/submit_sfno_mmd.sh
# ============================================================================

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields/FeatureMetric

# ── Config (override via env) ───────────────────────────────────────────────
export SFNO_REPO="${SFNO_REPO:-$HOME/SFNO-Embedding}"
ERA5_5VAR="${ERA5_5VAR:-data/era5_5var_2020.nc}"
GC_5VAR="${GC_5VAR:-data/graphcast_5var_2020_lead24h.nc}"
CHANNELS="${CHANNELS:-5}"        # smallest model
RES="${RES:-15}"                 # 15 -> 15x30
POOLING="${POOLING:-flatten}"    # concatenate the full latent
N_SAMPLES="${N_SAMPLES:-1000}"   # cap; uses all available if pool is smaller
SEED="${SEED:-0}"

mkdir -p plots

# ── Preflight: fail fast with clear guidance if prerequisites are missing ────
missing=0
for f in "$SFNO_REPO/weights/model_${CHANNELS}c_${RES}.pth" \
         "$SFNO_REPO/weights/static_fields.pth" \
         "$ERA5_5VAR" "$GC_5VAR"; do
    if [ ! -e "$f" ]; then
        echo "MISSING: $f"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo ""
    echo "Prerequisites missing — see the PREREQUISITES block in this script."
    echo "Stage the SFNO repo/weights and download the 5-var data on the login"
    echo "node, then re-submit."
    exit 1
fi

echo "Starting SFNO MMD eval on node: $(hostname)"
nvidia-smi || true
echo "SFNO_REPO=$SFNO_REPO"
echo "ERA5=$ERA5_5VAR  GraphCast=$GC_5VAR"
echo "config: ${CHANNELS}c ${RES}  pool=$POOLING  n=$N_SAMPLES"

python eval/eval_sfno_real_vs_forecast.py \
    --era5-path "$ERA5_5VAR" \
    --graphcast-path "$GC_5VAR" \
    --channels "$CHANNELS" \
    --res "$RES" \
    --pooling "$POOLING" \
    --mmd-only \
    --n-samples "$N_SAMPLES" \
    --batch-size 16 \
    --num-workers 2 \
    --seed "$SEED"

echo "SFNO MMD eval finished!"
