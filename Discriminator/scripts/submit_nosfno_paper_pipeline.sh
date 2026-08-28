#!/usr/bin/env bash
#SBATCH --job-name=nosfno-paper-pipeline
#SBATCH --account=pmlr_jobs
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_nosfno_paper_pipeline_%j.out
#SBATCH --error=slurm_nosfno_paper_pipeline_%j.err

# Complete no-SFNO workflow. Manuscript-sized PNG/PDF/NPZ bundles go to plots/paper/.
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
export PIPELINE_RUNS_DIR="${PIPELINE_RUNS_DIR:-/work/scratch/${USER}/baseline_pipeline_runs}"
cd "${REPO_DIR}"
python scripts/run_baseline_pipeline.py \
  "pipeline.id=nosfno-paper-${SLURM_JOB_ID:-local}" \
  "pipeline.stages=[fit_histogram_matching,fit_moment_matching,train_discriminators,evaluate_standard_metrics,evaluate_discriminator_metrics,plot]" \
  target_discriminator.sfno.enabled=false \
  baseline.discriminator.sfno.enabled=false \
  target_discriminator.train_attention_squeezenet=false \
  plotting.profile=paper \
  plotting.save_pdf=true \
  "pipeline.wandb.tags=[paper,nosfno]" \
  "$@"
