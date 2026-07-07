#!/bin/bash
#SBATCH --job-name=process_fast
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=process_fast_%j.out
#SBATCH --error=process_fast_%j.err

# Defaults assume the fast download outputs written by the GFS/GEFS fast sbatch
# wrappers. Override MODEL to process one dataset at a time.

set -eo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pmlr}"
WB2_REGRID_SCRIPT="${WB2_REGRID_SCRIPT:-/home/yelberkennou/weatherbench2/scripts/regrid.py}"
MODEL="${MODEL:-gfs}"
CONVERT_NETCDF="${CONVERT_NETCDF:-1}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${MODEL}" == "gfs" ]]; then
  INPUT_DIR="${INPUT_DIR:-/cluster/courses/pmlr/teams/team07/data/gfs_fast}"
  SCRATCH_DIR="${SCRATCH_DIR:-/work/scratch/yelberkennou/gfs_fast_processed}"
  OUTPUT_DIR="${OUTPUT_DIR:-/cluster/courses/pmlr/teams/team07/data/gfs_fast_processed}"
  START_DATE="${START_DATE:-20210501}"
  END_DATE="${END_DATE:-20231231}"
  CYCLES="${CYCLES:-00 06 12 18}"
  MEMBERS="${MEMBERS:-0}"
elif [[ "${MODEL}" == "gefs" ]]; then
  INPUT_DIR="${INPUT_DIR:-/cluster/courses/pmlr/teams/team07/data/gefs_fast}"
  SCRATCH_DIR="${SCRATCH_DIR:-/work/scratch/yelberkennou/gefs_fast_processed}"
  OUTPUT_DIR="${OUTPUT_DIR:-/cluster/courses/pmlr/teams/team07/data/gefs_fast_processed}"
  START_DATE="${START_DATE:-20201001}"
  END_DATE="${END_DATE:-20231231}"
  CYCLES="${CYCLES:-00}"
  MEMBERS="${MEMBERS:-0}"
else
  echo "MODEL must be gfs or gefs." >&2
  exit 2
fi

LEAD_HOURS="${LEAD_HOURS:-6 12 24 48 96 192}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "CONDA_SH does not exist: ${CONDA_SH}" >&2
  exit 2
fi

mkdir -p "${SCRATCH_DIR}" "${OUTPUT_DIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

cd "${REPO_DIR}"

MODEL="${MODEL}" \
INPUT_DIR="${INPUT_DIR}" \
SCRATCH_DIR="${SCRATCH_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
WB2_REGRID_SCRIPT="${WB2_REGRID_SCRIPT}" \
START_DATE="${START_DATE}" \
END_DATE="${END_DATE}" \
CYCLES="${CYCLES}" \
LEAD_HOURS="${LEAD_HOURS}" \
MEMBERS="${MEMBERS}" \
CONVERT_NETCDF="${CONVERT_NETCDF}" \
OVERWRITE="${OVERWRITE}" \
PYTHON=python \
bash download/process_fast_surface_forecasts.sh
