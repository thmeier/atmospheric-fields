#!/usr/bin/env bash
#SBATCH --job-name=paper5corr-fsplice
#SBATCH --account=pmlr_jobs
#SBATCH --time=02:00:00
#SBATCH --output=slurm_field_splice_composite_%j.out
#SBATCH --error=slurm_field_splice_composite_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${SUBMIT_DIR}/scripts/run_baseline_pipeline.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
else
  REPO_DIR="${SUBMIT_DIR}/Discriminator"
fi
source /etc/profile.d/modules.sh
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate pmlr

export DATA_DIR="${DATA_DIR:-/cluster/courses/pmlr/teams/team07/data}"
export PIPELINE_RUNS_DIR="${PIPELINE_RUNS_DIR:-/work/scratch/${USER}/baseline_pipeline_runs}"
VARIABLE_TAG="2m_temperature__10m_u_component_of_wind__10m_v_component_of_wind__mean_sea_level_pressure"
TEAM_DIR="${TEAM_DIR:-/cluster/courses/pmlr/teams/team07}"

# The original five-corruption paper run is authoritative. Dedicated reruns
# contribute only the field_splice rows that are absent from that run.
BASE_RUN="${BASE_RUN:-${TEAM_DIR}/results/paper-errorbars-5corr}"
STANDARD_RUN="${STANDARD_RUN:-${TEAM_DIR}/baseline_pipeline_runs/field-splice-baselines-10fold-20260828-021618}"
DISCRIMINATOR_RUN="${DISCRIMINATOR_RUN:-${TEAM_DIR}/baseline_pipeline_runs/field-splice-discriminator-eval-5fold-20260828-110819}"
COMPOSITE_ID="${COMPOSITE_ID:-paper-errorbars-5corr-plus-field-splice-$(date +%Y%m%d-%H%M%S)}"
COMPOSITE_RUN="${PIPELINE_RUNS_DIR}/${COMPOSITE_ID}"

cd "${REPO_DIR}"
if [[ -d "${COMPOSITE_RUN}/${VARIABLE_TAG}" ]]; then
  echo "Reusing existing composite data: ${COMPOSITE_RUN}/${VARIABLE_TAG}"
else
  python scripts/compose_corruption_plot_data.py \
    --base-run "${BASE_RUN}" \
    --standard-run "${STANDARD_RUN}" \
    --discriminator-run "${DISCRIMINATOR_RUN}" \
    --corruption field_splice \
    --output-dir "${COMPOSITE_RUN}/${VARIABLE_TAG}"
fi

python scripts/run_baseline_pipeline.py \
  "pipeline.id=${COMPOSITE_ID}" \
  pipeline.resume=true \
  "pipeline.runs_dir=${PIPELINE_RUNS_DIR}" \
  'pipeline.stages=[plot]' \
  'baseline.corruptions=[gaussian_blur,hf_noise,checkerboard_2px,hemisphere_splice,field_splice]' \
  target_discriminator.interpretability.enabled=false \
  baseline.discriminator.sfno.enabled=false \
  target_discriminator.sfno.enabled=false \
  plotting.profile=paper \
  plotting.save_pdf=true \
  pipeline.wandb.enabled=true \
  pipeline.wandb.mode=online \
  pipeline.wandb.project=weather-discriminator-baselines \
  'pipeline.wandb.tags=[paper,replot,authoritative-paper-errorbars-5corr,field-splice,temporal-resampling,nosfno]' \
  "$@"

echo "Composite paper run: ${COMPOSITE_RUN}"
