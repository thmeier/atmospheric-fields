#!/bin/bash
#SBATCH --job-name=aifs_surface
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=aifs_surface_%j.out
#SBATCH --error=aifs_surface_%j.err

# SLURM wrapper for download_aifs_surface_forecasts.sh.
#
# Defaults are set for the ETH PMLR cluster account/user paths below. Override
# with sbatch exports only when needed:
#   CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV_NAME=pmlr
#   WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py
#   TIME_START=2025-01-01T00
#   TIME_END=2025-12-31T18
#   MONTHS="202501 202502 ..."
#   LEAD_HOURS="6 12 24 48 96 192"
#   SCRATCH_DIR=/scratch/$USER/dynamical_native
#   OUTPUT_DIR=/cluster/courses/pmlr/teams/team07/data/dynamical
#   CONVERT_NETCDF=1|0
#   SKIP_EXISTING=1|0
#   KEEP_SOURCE_UNITS=1|0

set -eo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pmlr}"
TIME_START="${TIME_START:-2025-01-01T00}"
TIME_END="${TIME_END:-2025-12-31T18}"
MONTHS="${MONTHS:-202501 202502 202503 202504 202505 202506 202507 202508 202509 202510 202511 202512}"
LEAD_HOURS="${LEAD_HOURS:-6 12 24 48 96 192}"
SCRATCH_DIR="${SCRATCH_DIR:-/work/scratch/yelberkennou/dynamical_native}"
OUTPUT_DIR="${OUTPUT_DIR:-/cluster/courses/pmlr/teams/team07/data/dynamical}"
CONVERT_NETCDF="${CONVERT_NETCDF:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
KEEP_SOURCE_UNITS="${KEEP_SOURCE_UNITS:-0}"
WB2_REGRID_SCRIPT="${WB2_REGRID_SCRIPT:-/home/yelberkennou/weatherbench2/scripts/regrid.py}"

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

mkdir -p "${SCRATCH_DIR}" "${OUTPUT_DIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

cd "${REPO_DIR}"

RUNNER=conda \
PYTHON=python \
TIME_START="${TIME_START}" \
TIME_END="${TIME_END}" \
MONTHS="${MONTHS}" \
LEAD_HOURS="${LEAD_HOURS}" \
SCRATCH_DIR="${SCRATCH_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CONVERT_NETCDF="${CONVERT_NETCDF}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
KEEP_SOURCE_UNITS="${KEEP_SOURCE_UNITS}" \
WB2_REGRID_SCRIPT="${WB2_REGRID_SCRIPT}" \
bash download/download_aifs_surface_forecasts.sh
