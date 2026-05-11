#!/bin/bash
#SBATCH --job-name=era5_self_dist
#SBATCH --time=04:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=36G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=logs/era5_self_distance_%j.out
#SBATCH --error=logs/era5_self_distance_%j.err

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr
set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields
mkdir -p logs
mkdir -p plots/era5_self_distance

echo "Starting ERA5 self-distance evaluation on node: $(hostname)"
echo "Conda Env:" $(conda info --envs | grep -v '#' | grep '*' | awk '{print $1}')
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA built:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

PANGU_PATH="/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/pangu_surface_2020_lead24h.nc"
GRAPHCAST_PATH="/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/graphcast_surface_2020_lead24h.nc"

# ------ Phase 1: ERA5 bootstrap null distribution ----------------------------
# Splits ERA5 into two random halves 20 times and computes FID + MMD each time.
# Establishes the baseline variance of the metric under the null hypothesis
# (both inputs are real ERA5).
echo -e "\n--- Phase 1: ERA5 self-distance bootstrap (mean pooling) ---"
python eval/eval_era5_self_distance.py \
    --model both \
    --n-trials 20 \
    --n-per-split 250 \
    --batch-size 64 \
    --lazy

# ------ Phase 2: ERA5 vs Pangu / GraphCast -----------------------------------
# Compares the ERA5 self-distance baseline against the distance to 24 h
# forecasts from Pangu and GraphCast, using the same encoder and sample count.
echo -e "\n--- Phase 2: ERA5 vs forecasts comparison (mean pooling) ---"
python eval/eval_era5_self_distance.py \
    --model both \
    --n-trials 20 \
    --n-per-split 250 \
    --batch-size 64 \
    --lazy \
    --pangu-path "$PANGU_PATH" \
    --graphcast-path "$GRAPHCAST_PATH"

# ------ Phase 3: repeat with max pooling for comparison ----------------------
# The pooling strategy affects the latent geometry; max pooling may be more
# sensitive to local extremes.  Run both to see if the conclusion holds.
echo -e "\n--- Phase 3: ERA5 self-distance bootstrap (max pooling) ---"
EXTRACT_FEATURES_POOLING=max python eval/eval_era5_self_distance.py \
    --model both \
    --n-trials 20 \
    --n-per-split 250 \
    --batch-size 64 \
    --lazy

echo -e "\n--- Phase 4: ERA5 vs forecasts comparison (max pooling) ---"
EXTRACT_FEATURES_POOLING=max python eval/eval_era5_self_distance.py \
    --model both \
    --n-trials 20 \
    --n-per-split 250 \
    --batch-size 64 \
    --lazy \
    --pangu-path "$PANGU_PATH" \
    --graphcast-path "$GRAPHCAST_PATH"

echo -e "\nERA5 self-distance evaluation finished successfully!"
