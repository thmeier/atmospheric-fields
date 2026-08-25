#!/usr/bin/env bash
#SBATCH --job-name=paper-temporal
#SBATCH --account=pmlr_jobs
#SBATCH --partition=jobs
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --mem=48G
#SBATCH --output=slurm_paper_temporal_%j.out
#SBATCH --error=slurm_paper_temporal_%j.err

# Paper deliverables over the temporal-resampling protocol, without SFNO.
#
# Produces, under <run>/<variable-tag>/plots/:
#   corruption_by_type_normalized.png                 normalized metrics vs severity
#   discriminator/squeezenet/corruption_strength_reverse_kl.png
#   discriminator/squeezenet/lead_time_reverse_kl.png
#   target_logit_distributions/squeezenet/forecast/<model>/all_lead_times.png
# plus the blind-spot N/M table rendered from the same draws that band the curves.
#
# Environment: the pmlr conda env has torch but NO hydra, so the pipeline is run
# from the scratch venv that carries hydra over it with --system-site-packages.
# PROJ_DATA/PROJ_LIB point cartopy at the conda env's proj share. This mirrors
# download/submit_ucast_evalplot.sh, which is the last thing known to work here.
#
# Resources: this site rejects every per-task/per-node CPU request
# (--cpus-per-task, -c, --cpus-per-gpu, --gres all error out) and clamps each job
# to one GPU and three CPUs no matter what is asked for -- a job submitted with
# --gpus=8 still allocates cpu=3,gres/gpu=1. So the resample pool gets three
# workers, and WORKERS is read from nproc rather than guessed.
#
# --constraint=2080ti over the newer 5060ti: those nodes are sm_120 and need
# torch >= 2.7 / cu128, while this env ships torch 2.5.1+cu121.
#
# Output goes to /work/scratch (home has ~4 GB free), which is auto-cleaned every
# 1-7 days, so the small deliverables are copied back to $HOME at the end.
#
# Walltime is capped at 24h. The pipeline writes a per-stage manifest, so a run
# that does not finish continues with the same PIPELINE_ID and RESUME=true:
#   PIPELINE_ID=<id> RESUME=true sbatch scripts/submit_paper_temporal_pipeline.sh
set -eo pipefail

VENV="${VENV:-/work/scratch/ddemler/embedding_smoke/venv}"
PYTHON="${VENV}/bin/python"
[[ -x "${PYTHON}" ]] || { echo "No interpreter at ${PYTHON}" >&2; exit 2; }
export PROJ_DATA="${PROJ_DATA:-$HOME/miniconda3/envs/pmlr/share/proj}"
export PROJ_LIB="${PROJ_DATA}"
export DATA_DIR="${DATA_DIR:-/cluster/courses/pmlr/teams/team07/data}"
export PYTHONUNBUFFERED=1

OUT="${OUT:-/work/scratch/ddemler/paper_temporal}"
PIPELINE_ID="${PIPELINE_ID:-paper-temporal-${SLURM_JOB_ID:-local}}"
# One worker per genuinely allocated CPU. SLURM_CPUS_PER_TASK is unset here
# because the request form is rejected, so nproc is the only honest source.
# The pool pins each child to cpus/workers BLAS threads, which is also what makes
# a run reproducible: mmd_rbf shifts by ~2e-3 relative when that count changes.
WORKERS="${WORKERS:-$(nproc)}"
STAGES="${STAGES:-[train_discriminators,evaluate_standard_metrics,evaluate_discriminator_metrics,plot]}"
RESUME="${RESUME:-false}"

cd "$HOME/atmospheric-fields/Discriminator"
echo "node=$(hostname) workers=${WORKERS} pipeline=${PIPELINE_ID} resume=${RESUME}"
echo "stages=${STAGES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PYTHON}" scripts/run_baseline_pipeline.py \
  "output_dir=${OUT}" \
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
  "pipeline.wandb.enabled=true" \
  "pipeline.wandb.project=weather-discriminator-baselines" \
  "pipeline.wandb.entity=weather-realism-pmlr" \
  "pipeline.wandb.tags=[paper,nosfno,temporal-resampling]" \
  "$@"

RUN_DIR="$(find "${OUT}" -type d -name "${PIPELINE_ID}" -path '*pipeline_runs*' | head -1)"
[[ -n "${RUN_DIR}" ]] || { echo "Could not locate the run directory under ${OUT}" >&2; exit 0; }
DRAWS="$(find "${RUN_DIR}" -maxdepth 3 -name fixed_metric_draws.csv | head -1)"
if [[ -n "${DRAWS}" ]]; then
  BLINDSPOTS="${RUN_DIR}/blindspots"
  "${PYTHON}" scripts/blindspots_from_temporal_draws.py "${DRAWS}" --output-dir "${BLINDSPOTS}"
  "${PYTHON}" scripts/plot_bootstrap_blindspots.py "+bootstrap_null.data_dir=${BLINDSPOTS}"
fi

# Scratch is auto-cleaned every 1-7 days. Copy the small deliverables (figures,
# CSVs, LaTeX) home, but not the multi-GB NetCDF diagnostics or the per-resample
# child directories -- home only has a few GB free.
KEEP="$HOME/paper_temporal_results/${PIPELINE_ID}"
mkdir -p "${KEEP}"
rsync -a --exclude='*.nc' --exclude='resamples/' --exclude='models/' \
      "${RUN_DIR}/" "${KEEP}/" 2>/dev/null || \
  cp -r "${RUN_DIR}"/* "${KEEP}/" 2>/dev/null || true
echo
echo "Run directory : ${RUN_DIR}"
echo "Copied home to: ${KEEP}  ($(du -sh "${KEEP}" 2>/dev/null | cut -f1))"
echo "Home free     : $(df -h "$HOME" 2>/dev/null | tail -1 | awk '{print $4}')"
