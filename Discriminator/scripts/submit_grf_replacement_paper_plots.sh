#!/usr/bin/env bash
#SBATCH --job-name=grf-paper-refresh
#SBATCH --account=pmlr_jobs
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_grf_paper_refresh_%j.out
#SBATCH --error=slurm_grf_paper_refresh_%j.err

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
TEAM_ROOT="/cluster/courses/pmlr/teams/team07"
BASE_RUN="${BASE_RUN:-${TEAM_ROOT}/results/paper-temporal-106957}"
GRF_TRAIN_RUN="${GRF_TRAIN_RUN:-${TEAM_ROOT}/baseline_pipeline_runs/grf-only-5fold-20260827-181829}"
GRF_MAX_SEVERITY="${GRF_MAX_SEVERITY:-1.0}"
VARIABLE_TAG="2m_temperature__10m_u_component_of_wind__10m_v_component_of_wind__mean_sea_level_pressure"
JOB_TOKEN="${SLURM_JOB_ID:-local}"
STANDARD_ID="paper-temporal-grf-standard-0to1-${JOB_TOKEN}"
EVAL_ID="paper-temporal-grf-eval-0to1-${JOB_TOKEN}"
COMPOSITE_ID="paper-temporal-106957-grf-0to1-refresh-${JOB_TOKEN}"
CHECKPOINTS="${GRF_TRAIN_RUN}/${VARIABLE_TAG}/models/target_discriminators"
STANDARD_RUN="${PIPELINE_RUNS_DIR}/${STANDARD_ID}"
EVAL_RUN="${PIPELINE_RUNS_DIR}/${EVAL_ID}"
COMPOSITE_RUN="${PIPELINE_RUNS_DIR}/${COMPOSITE_ID}"

cd "${REPO_DIR}"
python scripts/run_baseline_pipeline.py \
  "pipeline.id=${STANDARD_ID}" \
  'pipeline.stages=[evaluate_standard_metrics]' \
  baseline.evaluate_forecasts=false \
  'baseline.corruptions=[grf]' \
  'baseline.metrics=[zonal_energy_spectrum_log_l2,global_mean_wasserstein,scwd,mmd_rbf]' \
  "+baseline.corruption_severity_max_overrides.grf=${GRF_MAX_SEVERITY}" \
  temporal_resampling.fixed_replicates=10 \
  temporal_resampling.learned_replicates=5 \
  'temporal_resampling.learned_test_windows=[[5,11],[9,15],[13,19],[17,23],[20,26]]' \
  'pipeline.wandb.tags=[paper,grf,replacement,standard-metrics,0to1]'

python scripts/run_baseline_pipeline.py \
  "pipeline.id=${EVAL_ID}" \
  'pipeline.stages=[evaluate_discriminator_metrics]' \
  "pipeline.input_checkpoint_dir=${CHECKPOINTS}" \
  'baseline.corruptions=[grf]' \
  'baseline.discriminator.forecast_files={}' \
  "+baseline.discriminator.corruption_severity_max_overrides.grf=${GRF_MAX_SEVERITY}" \
  baseline.discriminator.sfno.enabled=false \
  baseline.discriminator.attention_squeezenet.enabled=false \
  baseline.discriminator.equator_masked_hemisphere_splice.enabled=false \
  temporal_resampling.learned_replicates=5 \
  'temporal_resampling.learned_test_windows=[[5,11],[9,15],[13,19],[17,23],[20,26]]' \
  'pipeline.wandb.tags=[paper,grf,replacement,discriminator-evaluation,0to1]'

python scripts/compose_grf_plot_data.py \
  --base-run "${BASE_RUN}" \
  --grf-standard-run "${STANDARD_RUN}" \
  --grf-discriminator-run "${EVAL_RUN}" \
  --output-dir "${COMPOSITE_RUN}"

python scripts/render_npz_paper_plots.py \
  "${BASE_RUN}/${VARIABLE_TAG}/plots/paper" \
  --derived-source "${COMPOSITE_RUN}" \
  --exclude-grf-bundles \
  --output-dir "${COMPOSITE_RUN}" \
  --wandb \
  --wandb-project weather-discriminator-baselines \
  --wandb-entity weather-realism-pmlr \
  --wandb-name "${COMPOSITE_ID}/paper-gallery"

echo "Composite paper gallery: ${COMPOSITE_RUN}"
