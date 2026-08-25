#!/usr/bin/env bash
#SBATCH --job-name=bootstrap-null
#SBATCH --account=pmlr_jobs
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_bootstrap_null_%j.out
#SBATCH --error=slurm_bootstrap_null_%j.err

# Resample the ERA5 null distribution, evaluate corruption curves, and render bootstrap blind-spot plots.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${SUBMIT_DIR}/scripts/run_baseline_pipeline.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/Discriminator/scripts/run_baseline_pipeline.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}/Discriminator"
else
  echo "Could not locate Discriminator/scripts/run_baseline_pipeline.py from ${SUBMIT_DIR}" >&2
  exit 2
fi
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pmlr}"
source /etc/profile.d/modules.sh
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1
export DATA_DIR="${DATA_DIR:-/cluster/courses/pmlr/teams/team07/data}"
cd "${REPO_DIR}"
python scripts/run_baseline_pipeline.py \
  "pipeline.id=bootstrap-null-${SLURM_JOB_ID:-local}" \
  "pipeline.stages=[evaluate_bootstrap_null,plot_bootstrap_blindspots]" \
  "pipeline.wandb.tags=[bootstrap-null,blindspots]" \
  "$@"
