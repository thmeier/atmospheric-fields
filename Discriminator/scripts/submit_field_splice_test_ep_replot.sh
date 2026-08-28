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
BASE_RUN="${PIPELINE_RUNS_DIR}/paper-errorbars-5corr-test-ep-replot-20260828"
STANDARD_RUN="${PIPELINE_RUNS_DIR}/field-splice-baselines-10fold-20260828-021618"
DISCRIMINATOR_RUN="${PIPELINE_RUNS_DIR}/field-splice-test-ep-5fold-20260828"
COMPOSITE_ID="paper-errorbars-5corr-plus-field-splice-test-ep-20260828"
COMPOSITE_RUN="${PIPELINE_RUNS_DIR}/${COMPOSITE_ID}"

cd "${REPO_DIR}"
python scripts/compose_corruption_plot_data.py \
  --base-run "${BASE_RUN}" \
  --standard-run "${STANDARD_RUN}" \
  --discriminator-run "${DISCRIMINATOR_RUN}" \
  --corruption field_splice \
  --output-dir "${COMPOSITE_RUN}/${VARIABLE_TAG}"

python scripts/run_baseline_pipeline.py \
  "pipeline.id=${COMPOSITE_ID}" \
  pipeline.resume=true \
  "pipeline.runs_dir=${PIPELINE_RUNS_DIR}" \
  'pipeline.stages=[plot]' \
  'baseline.corruptions=[gaussian_blur,hemisphere_splice,checkerboard_2px,grf,hf_noise,field_splice]' \
  target_discriminator.interpretability.enabled=false \
  baseline.discriminator.sfno.enabled=false \
  target_discriminator.sfno.enabled=false \
  plotting.profile=paper \
  plotting.save_pdf=true \
  pipeline.wandb.enabled=true \
  pipeline.wandb.mode=online \
  pipeline.wandb.project=weather-discriminator-baselines \
  'pipeline.wandb.tags=[paper,replot,test-p-expectation,field-splice,temporal-resampling,nosfno]'

echo "Composite paper run: ${COMPOSITE_RUN}"
