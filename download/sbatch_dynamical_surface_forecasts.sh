#!/bin/bash
#SBATCH --job-name=dyn_surface
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=dyn_surface_%j.out
#SBATCH --error=dyn_surface_%j.err

# SLURM wrapper for download_dynamical_surface_forecasts.sh.
#
# Required sbatch exports:
#   CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV_NAME=pmlr
#   WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py
#
# Optional sbatch exports:
#   MODE=all|gfs|gefs
#   SCRATCH_DIR=/scratch/$USER/dynamical_native
#   OUTPUT_DIR=/cluster/courses/pmlr/teams/team07/data/dynamical
#   CONVERT_NETCDF=1|0
#   GEFS_ENSEMBLE_MEMBERS=all|0|0,1,...

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
MODE="${MODE:-all}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/${USER}/dynamical_native}"
OUTPUT_DIR="${OUTPUT_DIR:-/cluster/courses/pmlr/teams/team07/data/dynamical}"
CONVERT_NETCDF="${CONVERT_NETCDF:-1}"
GEFS_ENSEMBLE_MEMBERS="${GEFS_ENSEMBLE_MEMBERS:-all}"
WB2_REGRID_SCRIPT="${WB2_REGRID_SCRIPT:-}"

if [[ -z "${CONDA_SH}" ]]; then
  echo "Set CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh" >&2
  exit 2
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "CONDA_SH does not exist: ${CONDA_SH}" >&2
  exit 2
fi
if [[ -z "${CONDA_ENV_NAME}" ]]; then
  echo "Set CONDA_ENV_NAME, e.g. CONDA_ENV_NAME=pmlr" >&2
  exit 2
fi
if [[ -z "${WB2_REGRID_SCRIPT}" ]]; then
  echo "Set WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py" >&2
  exit 2
fi

mkdir -p "${REPO_DIR}/logs" "${SCRATCH_DIR}" "${OUTPUT_DIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

cd "${REPO_DIR}"

RUNNER=conda \
PYTHON=python \
MODE="${MODE}" \
SCRATCH_DIR="${SCRATCH_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CONVERT_NETCDF="${CONVERT_NETCDF}" \
GEFS_ENSEMBLE_MEMBERS="${GEFS_ENSEMBLE_MEMBERS}" \
WB2_REGRID_SCRIPT="${WB2_REGRID_SCRIPT}" \
bash download/download_dynamical_surface_forecasts.sh
