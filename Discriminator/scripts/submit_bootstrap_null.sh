#!/usr/bin/env bash
#SBATCH --job-name=bootstrap_null
#SBATCH --account=pmlr_jobs
#SBATCH --time=05:00:00
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --mem=48G
#SBATCH --output=/work/scratch/ddemler/bootstrap_null/slurm_%j.out
#SBATCH --error=/work/scratch/ddemler/bootstrap_null/slurm_%j.err

# Resample the ERA5 null behind the corruption blind-spot table, sweep the
# severity ladder with paired resampling, and render the figure and tables.
#
#   sbatch scripts/submit_bootstrap_null.sh
#   sbatch scripts/submit_bootstrap_null.sh pipeline.wandb.mode=offline
#
# Arguments after the script name pass through as Hydra overrides.
#
# Runs through run_baseline_pipeline.py so both stages are tracked in W&B.
# W&B online mode calls wandb.login() and raises on failure; credentials come
# from ~/.netrc on this cluster. Pass pipeline.wandb.mode=offline if that is
# ever unavailable.
#
# The pmlr conda env has no hydra, hence the scratch venv, which is built with
# --system-site-packages over it. PROJ_DATA has to be pointed at the conda env
# by hand because cartopy resolves it through the activated environment.

set -euo pipefail

export PROJ_DATA="${PROJ_DATA:-${HOME}/miniconda3/envs/pmlr/share/proj}"
export PROJ_LIB="${PROJ_DATA}"
export GDAL_DATA="${GDAL_DATA:-${HOME}/miniconda3/envs/pmlr/share/gdal}"
export DATA_DIR="${DATA_DIR:-/cluster/courses/pmlr/teams/team07/data}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

PYTHON="${BOOTSTRAP_PYTHON:-/work/scratch/ddemler/embedding_smoke/venv/bin/python}"
OUTPUT_DIR="${BOOTSTRAP_OUTPUT_DIR:-/work/scratch/ddemler/bootstrap_null}"

cd "${HOME}/atmospheric-fields/Discriminator"
echo "bootstrap null job ${SLURM_JOB_ID:-interactive} on $(hostname)"
echo "DATA_DIR=${DATA_DIR}  OUTPUT_DIR=${OUTPUT_DIR}"
nvidia-smi || true

"${PYTHON}" scripts/run_baseline_pipeline.py \
    "pipeline.stages=[evaluate_bootstrap_null,plot_bootstrap_blindspots]" \
    "output_dir=${OUTPUT_DIR}" \
    "$@"
