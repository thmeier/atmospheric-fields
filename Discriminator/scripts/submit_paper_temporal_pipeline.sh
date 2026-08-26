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
# Working output goes to /work/scratch (100 GB). Durable copies go to the shared
# team07 directory on /cluster/courses, NOT to $HOME -- home is a 20 GB quota that
# is already ~17 GB full. team07 has terabytes free and inherits ACLs that let the
# whole team read the results.
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
# Single-run defaults: one critic-training window and one resample draw, so every
# curve is a bare point estimate with no bands. This is the "final plots first"
# pass over the full 4-field input; error bars come later, only for the handful of
# corruptions worth the compute. To restore the banded protocol, override:
#   LEARNED_REPLICATES=5 LEARNED_WINDOWS='[[5,11],[9,15],[13,19],[17,23],[20,26]]' \
#   FIXED_REPLICATES=25 sbatch scripts/submit_paper_temporal_pipeline.sh
# fixed_replicates must stay >= learned_replicates, and learned_replicates must
# equal the number of learned_test_windows.
LEARNED_REPLICATES="${LEARNED_REPLICATES:-1}"
FIXED_REPLICATES="${FIXED_REPLICATES:-1}"
LEARNED_WINDOWS="${LEARNED_WINDOWS:-[[20,26]]}"

cd "$HOME/atmospheric-fields/Discriminator"

# The shipped SWIFT export is zlib-compressed with chunks spanning 244 timesteps,
# so a single-field read decompresses ~57 MB. Every stage here reads one
# (time, lead) field at a time in shuffled order, which made SWIFT train at
# 7.8 s/step against GraphCast's 0.16 s/step. A contiguous, uncompressed rewrite
# is byte-identical and reads 154x faster (123 ms -> 0.8 ms per field). Rebuild it
# on scratch when absent -- scratch is auto-cleaned, and the shared team copy is
# deliberately left untouched.
FAST_DATA="${FAST_DATA:-/work/scratch/ddemler/data}"
SWIFT_NAME=swift_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc
if [[ ! -f "${FAST_DATA}/${SWIFT_NAME}" ]]; then
  echo "Rebuilding a contiguous SWIFT copy under ${FAST_DATA}"
  mkdir -p "${FAST_DATA}"
  "${PYTHON}" - "${DATA_DIR}/${SWIFT_NAME}" "${FAST_DATA}/${SWIFT_NAME}" <<'PYEOF'
import sys, xarray as xr
source, target = sys.argv[1], sys.argv[2]
ds = xr.open_dataset(source)
ds.to_netcdf(target, engine="netcdf4", encoding={
    v: {"zlib": False, "complevel": 0, "contiguous": True} for v in ds.data_vars})
ds.close()
print(f"wrote {target}")
PYEOF
fi

echo "node=$(hostname) workers=${WORKERS} pipeline=${PIPELINE_ID} resume=${RESUME}"
echo "stages=${STAGES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PYTHON}" scripts/run_baseline_pipeline.py \
  "output_dir=${OUT}" \
  "pipeline.id=${PIPELINE_ID}" \
  "pipeline.stages=${STAGES}" \
  "pipeline.resume=${RESUME}" \
  "temporal_resampling.workers=${WORKERS}" \
  "temporal_resampling.learned_replicates=${LEARNED_REPLICATES}" \
  "temporal_resampling.fixed_replicates=${FIXED_REPLICATES}" \
  "temporal_resampling.learned_test_windows=${LEARNED_WINDOWS}" \
  target_discriminator.sfno.enabled=false \
  baseline.discriminator.sfno.enabled=false \
  target_discriminator.train_attention_squeezenet=false \
  "baseline.corruptions=[hemisphere_splice,checkerboard_2px,equatorial_checker_texture,zonal_scanlines,meridional_scanlines,gaussian_blur,grf,hf_noise,pixel_replace]" \
  "baseline.forecast_files.SWIFT=[${FAST_DATA}/${SWIFT_NAME}]" \
  plotting.profile=paper \
  plotting.save_pdf=true \
  "pipeline.wandb.enabled=true" \
  "pipeline.wandb.project=weather-discriminator-baselines" \
  "pipeline.wandb.entity=weather-realism-pmlr" \
  "pipeline.wandb.tags=[paper,nosfno,temporal-resampling]" \
  "$@"

RUN_DIR="$(find "${OUT}" -type d -name "${PIPELINE_ID}" -path '*pipeline_runs*' | head -1)"
[[ -n "${RUN_DIR}" ]] || { echo "Could not locate the run directory under ${OUT}" >&2; exit 0; }
# The blind-spot table thresholds each metric against a p95 of the resampled null,
# which needs at least 20 null draws (NULL_REPLICATE_FLOOR). A single-draw "final
# plots first" run cannot produce it -- that is deferred to the later selective
# multi-sampling run -- so skip it here rather than let it abort the script (and
# with it the team07 sync) under `set -e`.
NULL_FLOOR=20
DRAWS="$(find "${RUN_DIR}" -maxdepth 3 -name fixed_metric_draws.csv | head -1)"
if [[ -n "${DRAWS}" && "${FIXED_REPLICATES:-1}" -ge "${NULL_FLOOR}" ]]; then
  BLINDSPOTS="${RUN_DIR}/blindspots"
  "${PYTHON}" scripts/blindspots_from_temporal_draws.py "${DRAWS}" --output-dir "${BLINDSPOTS}"
  "${PYTHON}" scripts/plot_bootstrap_blindspots.py "+bootstrap_null.data_dir=${BLINDSPOTS}"
else
  echo "Skipping blind-spot table: fixed_replicates=${FIXED_REPLICATES:-1} < ${NULL_FLOOR} (p95 null needs >=${NULL_FLOOR} draws)."
fi

# Keep everything worth reusing on the shared volume. Trained critics are ~2.8 MB
# each (~250 MB for all 90) and are what makes a rerun cheap: pass the copied tree
# back as temporal_resampling.input_run_dir to evaluate or replot without
# retraining. We also keep every per-resample metric artifact (csv/csv.gz/nc), so
# this backup is a COMPLETE resume source: with the same PIPELINE_ID and seed a
# later run can raise fixed_replicates (0..9 are reused byte-for-byte, only the
# new draws compute) or append learned windows without redoing finished work.
# Only the bulky per-fold plot trees (png/pdf) are left behind -- the canonical
# fold's plots are already copied up to the parent.
KEEP="${KEEP_ROOT:-/cluster/courses/pmlr/teams/team07/results}/${PIPELINE_ID}"
mkdir -p "${KEEP}"
rsync -a --prune-empty-dirs \
      --include='*/' --include='model.pth' --include='*.json' \
      --include='*.csv' --include='*.csv.gz' --include='*.nc' --exclude='*' \
      "${RUN_DIR}/resamples/" "${KEEP}/resamples/" 2>/dev/null || true
rsync -a --exclude='resamples/' "${RUN_DIR}/" "${KEEP}/" 2>/dev/null || \
  cp -r "${RUN_DIR}"/* "${KEEP}/" 2>/dev/null || true
echo "  critics kept: $(find "${KEEP}" -name model.pth 2>/dev/null | wc -l)"
echo
echo "Run directory : ${RUN_DIR}"
echo "Kept in team07: ${KEEP}  ($(du -sh "${KEEP}" 2>/dev/null | cut -f1))"
