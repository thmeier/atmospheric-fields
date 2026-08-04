#!/usr/bin/env bash
#SBATCH --job-name=baseline_pipeline
#SBATCH --account=pmlr_jobs
#SBATCH --time=05:00:00
#SBATCH --output=slurm_baseline_pipeline_%j.out
#SBATCH --error=slurm_baseline_pipeline_%j.err

# --time=4-00:00
# Submit the complete tracked baseline workflow non-interactively:
#
#   sbatch scripts/submit_baseline_pipeline.sh
#   sbatch scripts/submit_baseline_pipeline.sh pipeline.wandb.mode=offline
#   sbatch scripts/submit_baseline_pipeline.sh \
#       'pipeline.stages=[evaluate_standard_metrics,evaluate_discriminator_metrics,plot]'
#
# Any arguments after the script name are passed unchanged as Hydra overrides.
# Set DATA_DIR, SFNO_REPO, CONDA_SH, or CONDA_ENV_NAME before submitting to
# override these shared defaults. W&B must already be authenticated for the
# default online mode; use pipeline.wandb.mode=offline for an offline run.

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

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda activation script not found: ${CONDA_SH}" >&2
  exit 2
fi

source /etc/profile.d/modules.sh
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

export PYTHONUNBUFFERED=1
export DATA_DIR="${DATA_DIR:-/cluster/courses/pmlr/teams/team07/data}"
export SFNO_REPO="${SFNO_REPO:-${HOME}/SFNO-Embedding}"

cd "${REPO_DIR}"

echo "Baseline pipeline job ${SLURM_JOB_ID:-unknown} on $(hostname)"
echo "DATA_DIR=${DATA_DIR}"
echo "SFNO_REPO=${SFNO_REPO}"
nvidia-smi || true

python scripts/run_baseline_pipeline.py "$@"
