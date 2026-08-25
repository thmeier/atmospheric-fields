#!/usr/bin/env bash
#SBATCH --job-name=paper-temporal
#SBATCH --account=pmlr_jobs
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --mem=64G
#SBATCH --output=slurm_paper_temporal_%j.out
#SBATCH --error=slurm_paper_temporal_%j.err

# Paper deliverables over the temporal-resampling protocol, without SFNO.
#
# Produces, under <run>/<variable-tag>/plots/:
#   corruption_by_type_normalized.png                 normalized metrics vs severity
#   discriminator/squeezenet/corruption_strength_reverse_kl.png
#   discriminator/squeezenet/lead_time_reverse_kl.png
#   target_logit_distributions/squeezenet/forecast/<model>/all_lead_times.png
# and, after the follow-up command this script prints, the blind-spot N/M table.
#
# Resources: this site rejects every per-task/per-node CPU request
# (--cpus-per-task, -c, --cpus-per-gpu, --gres all error out) and clamps each job
# to one GPU and three CPUs no matter what is asked for -- a job submitted with
# --gpus=8 still allocates cpu=3,gres/gpu=1. So the resample pool gets three
# workers, not the dozens a full node would allow, and WORKERS is read from nproc
# rather than guessed.
#
# --constraint=2080ti over the newer 5060ti: those nodes are sm_120 and need
# torch >= 2.7 / cu128, while the pmlr env ships torch 2.5.1+cu121. Their higher
# core count is moot given the three-CPU clamp.
#
# Walltime is capped at 24h for pmlr_jobs. The pipeline writes a per-stage
# manifest, so a run that does not finish can be continued with the same
# PIPELINE_ID and RESUME=true rather than restarted.
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

PIPELINE_ID="${PIPELINE_ID:-paper-temporal-${SLURM_JOB_ID:-local}}"
# One worker per genuinely allocated CPU. SLURM_CPUS_PER_TASK is unset here
# because the request form is rejected, so nproc is the only honest source.
# The pool pins each child to cpus/workers BLAS threads, which is also what makes
# a run reproducible: mmd_rbf shifts by ~2e-3 relative when that count changes.
WORKERS="${WORKERS:-$(nproc)}"
STAGES="${STAGES:-[train_discriminators,evaluate_standard_metrics,evaluate_discriminator_metrics,plot]}"
RESUME="${RESUME:-false}"
echo "pipeline=${PIPELINE_ID} workers=${WORKERS} stages=${STAGES} resume=${RESUME}"

python scripts/run_baseline_pipeline.py \
  "pipeline.id=${PIPELINE_ID}" \
  "pipeline.stages=${STAGES}" \
  "pipeline.resume=${RESUME}" \
  "temporal_resampling.workers=${WORKERS}" \
  target_discriminator.sfno.enabled=false \
  baseline.discriminator.sfno.enabled=false \
  target_discriminator.train_attention_squeezenet=false \
  "baseline.corruptions=[hemisphere_splice,checkerboard_2px,equatorial_checker_texture,zonal_scanlines,meridional_scanlines,gaussian_blur,grf,hf_noise,pixel_replace]" \
  plotting.profile=paper \
  plotting.save_pdf=true \
  "pipeline.wandb.tags=[paper,nosfno,temporal-resampling]" \
  "$@"

RUN_DIR="$(python - "${PIPELINE_ID}" <<'PYTHON'
import sys
from pathlib import Path
matches = sorted(Path("results").rglob(f"pipeline_runs/{sys.argv[1]}"))
print(matches[-1] if matches else "", end="")
PYTHON
)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Pipeline finished but its run directory could not be located; render the table by hand." >&2
  exit 0
fi
DRAWS="$(find "${RUN_DIR}" -maxdepth 3 -name fixed_metric_draws.csv | head -1)"
BLINDSPOTS="${RUN_DIR}/blindspots"
python scripts/blindspots_from_temporal_draws.py "${DRAWS}" --output-dir "${BLINDSPOTS}"
python scripts/plot_bootstrap_blindspots.py "+bootstrap_null.data_dir=${BLINDSPOTS}"
echo
echo "Deliverables under ${RUN_DIR}"
echo "  blind-spot table : ${BLINDSPOTS}/blindspot_table.tex"
echo "  exceedance table : ${BLINDSPOTS}/null_exceedance_table.tex"
echo "  raw metric draws : ${DRAWS}"
