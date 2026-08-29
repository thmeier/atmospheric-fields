#!/usr/bin/env bash
#SBATCH --job-name=wandb-plot-pdfs
#SBATCH --account=pmlr_jobs
#SBATCH --time=00:20:00
#SBATCH --output=slurm_wandb_plot_upload_%j.out
#SBATCH --error=slurm_wandb_plot_upload_%j.err

set -euo pipefail
source /etc/profile.d/modules.sh
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate pmlr
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export WANDB_DIR="${WANDB_DIR:-/tmp/weather-discriminator-wandb-upload-${SLURM_JOB_ID}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/weather-discriminator-wandb-cache-${SLURM_JOB_ID}}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/tmp/weather-discriminator-wandb-data-${SLURM_JOB_ID}}"
python scripts/upload_plot_directory_to_wandb.py "$@"
