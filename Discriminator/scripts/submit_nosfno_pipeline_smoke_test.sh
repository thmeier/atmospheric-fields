#!/usr/bin/env bash
#SBATCH --job-name=nosfno-smoke
#SBATCH --account=pmlr_jobs
#SBATCH --time=00:30:00
#SBATCH --output=slurm_nosfno_smoke_%j.out
#SBATCH --error=slurm_nosfno_smoke_%j.err

# Small end-to-end validation: two learned folds, three fixed resamples, no SFNO.
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
  "pipeline.id=nosfno-smoke-${SLURM_JOB_ID:-local}" \
  "pipeline.stages=[train_discriminators,evaluate_standard_metrics,evaluate_discriminator_metrics,plot]" \
  temporal_resampling.learned_replicates=2 \
  temporal_resampling.fixed_replicates=3 \
  "temporal_resampling.learned_test_windows=[[5,11],[20,26]]" \
  "~baseline.forecast_files" \
  '+baseline.forecast_files.GraphCast=[${data_dir}/graphcast_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc]' \
  target_discriminator.sfno.enabled=false \
  baseline.discriminator.sfno.enabled=false \
  target_discriminator.train_attention_squeezenet=false \
  target_discriminator.equator_masked_hemisphere_splice.enabled=false \
  target_discriminator.epochs=1 \
  target_discriminator.max_train_samples=32 \
  target_discriminator.max_eval_samples=32 \
  target_discriminator.batch_size=8 \
  "baseline.corruptions=[gaussian_blur,field_splice]" \
  baseline.corruption_steps=2 \
  baseline.corruption_eval_samples=32 \
  baseline.eval_samples=32 \
  "baseline.metrics=[mean_bias,std_ratio_error]" \
  plotting.profile=dashboard \
  plotting.save_pdf=false \
  "pipeline.wandb.tags=[smoke,nosfno,temporal-resampling]" \
  "$@"
