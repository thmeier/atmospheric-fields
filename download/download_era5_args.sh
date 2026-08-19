#!/bin/bash
#SBATCH --account=pmlr_jobs
#SBATCH --job-name=download_wb2
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=wb2_download_%j.out
#SBATCH --error=wb2_download_%j.err

# Download monthly WeatherBench2 Zarr selections to NetCDF.
# This script does not regrid. Choose an already suitable WeatherBench2 source,
# e.g. a 240x121 1.5-degree store when WeatherBench-like data is required.
#
# Positional compatibility:
#   bash download/download_era5_args.sh DATASET_PATH TIME_START TIME_END TAG
#
# Environment overrides:
#   SOURCE_PATH=era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr
#   TIME_START=2020-01-01
#   TIME_END=2020-12-31
#   TAG=era5-gt
#   MONTHS="202001 202002 ..."
#   VARIABLES="2m_temperature 10m_u_component_of_wind ..."
#   LEVEL=850
#   LEAD_HOURS="6 12 24 48 96 192" | all
#   OUTPUT_DIR=/cluster/courses/pmlr/teams/team07/data
#   OUTPUT_FORMAT=netcdf|zarr
#   SKIP_EXISTING=1|0
#   CONDA_ENV_NAME=pmlr
#   PYTHON=python
#   DRY_RUN=1|0

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# See note in download_gen_data.sh: under sbatch the script runs from a spool dir,
# so resolve back to the repo checkout when the sibling worker is not alongside.
if [[ ! -f "${SCRIPT_DIR}/download_era5_netcdf.py" ]]; then
  SCRIPT_DIR="${DOWNLOAD_DIR:-${HOME}/atmospheric-fields/download}"
fi
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_PATH="${1:-${SOURCE_PATH:-era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr}}"
TIME_START="${2:-${TIME_START:-2020-01-01}}"
TIME_END="${3:-${TIME_END:-2020-12-31}}"
TAG="${4:-${TAG:-era5-gt}}"
MONTHS="${MONTHS:-}"
VARIABLES="${VARIABLES:-2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind mean_sea_level_pressure}"
LEVEL="${LEVEL:-}"
LEAD_HOURS="${LEAD_HOURS:-6 12 24 48 96 192}"
OUTPUT_DIR="${OUTPUT_DIR:-/cluster/courses/pmlr/teams/team07/data}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-netcdf}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pmlr}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${SOURCE_PATH}" == gs://* || "${SOURCE_PATH}" == /* ]]; then
  SOURCE="${SOURCE_PATH}"
else
  SOURCE="gs://weatherbench2/datasets/${SOURCE_PATH}"
fi

month_start() {
  local month="$1"
  echo "${month:0:4}-${month:4:2}-01"
}

month_end() {
  local month="$1"
  date -u -d "${month:0:4}-${month:4:2}-01 1 month -1 second" +"%Y-%m-%dT%H:%M:%S"
}

max_time() {
  if [[ "$1" > "$2" ]]; then
    echo "$1"
  else
    echo "$2"
  fi
}

min_time() {
  if [[ "$1" < "$2" ]]; then
    echo "$1"
  else
    echo "$2"
  fi
}

generate_months() {
  local start_month
  local end_month
  local current

  start_month="$(date -u -d "${TIME_START:0:7}-01" +"%Y%m")"
  end_month="$(date -u -d "${TIME_END:0:7}-01" +"%Y%m")"
  current="${start_month}"

  while [[ "${current}" -le "${end_month}" ]]; do
    echo "${current}"
    current="$(date -u -d "${current:0:4}-${current:4:2}-01 1 month" +"%Y%m")"
  done
}

resolve_months() {
  if [[ -n "${MONTHS}" ]]; then
    echo "${MONTHS}"
  else
    generate_months
  fi
}

run_python() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

download_month() {
  local month="$1"
  local start
  local end
  local output_file
  local level_args=()

  start="$(max_time "${TIME_START}" "$(month_start "${month}")")"
  end="$(min_time "${TIME_END}" "$(month_end "${month}")")"

  if [[ "${start}" > "${end}" ]]; then
    echo "Skipping ${month}; outside requested range ${TIME_START} through ${TIME_END}."
    return
  fi

  if [[ "${OUTPUT_FORMAT}" == "zarr" ]]; then
    output_file="${OUTPUT_DIR}/${TAG}_6steps_1.5deg_${month}.zarr"
  else
    output_file="${OUTPUT_DIR}/${TAG}_6steps_1.5deg_${month}.nc"
  fi
  # -e not -f: a zarr store is a directory, not a regular file.
  if [[ "${SKIP_EXISTING}" == "1" && -e "${output_file}" ]]; then
    echo "Skipping ${month}; output already exists: ${output_file}"
    return
  fi

  if [[ -n "${LEVEL}" ]]; then
    level_args=(--level ${LEVEL})
  fi

  run_python "${PYTHON}" "${SCRIPT_DIR}/download_era5_netcdf.py" \
    "${SOURCE}" \
    "${output_file}" \
    --time-start "${start}" \
    --time-end "${end}" \
    --variables ${VARIABLES} \
    --lead-hours "${LEAD_HOURS}" \
    --format "${OUTPUT_FORMAT}" \
    "${level_args[@]}"
}

mkdir -p "${OUTPUT_DIR}"

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "CONDA_SH does not exist: ${CONDA_SH}" >&2
    exit 2
  fi
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_DIR}"

for month in $(resolve_months); do
  download_month "${month}"
done
