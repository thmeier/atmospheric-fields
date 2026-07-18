#!/bin/bash
# Download native Dynamical ECMWF AIFS Single surface forecasts month by month,
# write native and regridded Zarr intermediates to scratch, and optionally
# convert the regridded Zarr to final NetCDF in OUTPUT_DIR. Defaults cover init
# times [2025, 2026).
#
# Required:
#   WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py
#
# Optional:
#   SCRATCH_DIR=data/dynamical/scratch
#   OUTPUT_DIR=data/dynamical/regridded
#   TIME_START=2025-01-01T00
#   TIME_END=2025-12-31T18
#   MONTHS="202501 202502 ..."
#   LEAD_HOURS="6 12 24 48 96 192"
#   CONVERT_NETCDF=1|0
#   SKIP_EXISTING=1|0
#   KEEP_SOURCE_UNITS=1|0
#   PYTHON=python
#   RUNNER=conda|uv

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCRATCH_DIR="${SCRATCH_DIR:-${REPO_DIR}/data/dynamical/scratch}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/data/dynamical/regridded}"
TIME_START="${TIME_START:-2025-01-01T00}"
TIME_END="${TIME_END:-2025-12-31T18}"
MONTHS="${MONTHS:-202501 202502 202503 202504 202505 202506 202507 202508 202509 202510 202511 202512}"
LEAD_HOURS="${LEAD_HOURS:-6 12 24 48 96 192}"
CONVERT_NETCDF="${CONVERT_NETCDF:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
KEEP_SOURCE_UNITS="${KEEP_SOURCE_UNITS:-0}"
PYTHON="${PYTHON:-python}"
RUNNER="${RUNNER:-conda}"
WB2_REGRID_SCRIPT="${WB2_REGRID_SCRIPT:-}"

UV_DATA_DEPS=(
  --with dynamical-catalog
  --with xarray
  --with zarr
  --with dask
  --with netcdf4
)

UV_REGRID_DEPS=(
  --with git+https://github.com/google-research/weatherbench2
  --with xarray
  --with zarr
  --with dask
  --with netcdf4
)

run_data_python() {
  if [[ "${RUNNER}" == "uv" ]]; then
    uv run "${UV_DATA_DEPS[@]}" python "$@"
  elif [[ "${RUNNER}" == "conda" ]]; then
    "${PYTHON}" "$@"
  else
    echo "Unsupported RUNNER=${RUNNER}. Use conda or uv." >&2
    exit 2
  fi
}

run_regrid_python() {
  local wb2_repo_dir
  wb2_repo_dir="$(cd "$(dirname "${WB2_REGRID_SCRIPT}")/.." && pwd)"

  if [[ "${RUNNER}" == "uv" ]]; then
    PYTHONPATH="${wb2_repo_dir}:${PYTHONPATH:-}" uv run "${UV_REGRID_DEPS[@]}" python "$@"
  elif [[ "${RUNNER}" == "conda" ]]; then
    PYTHONPATH="${wb2_repo_dir}:${PYTHONPATH:-}" "${PYTHON}" "$@"
  else
    echo "Unsupported RUNNER=${RUNNER}. Use conda or uv." >&2
    exit 2
  fi
}

require_regrid_script() {
  if [[ -z "${WB2_REGRID_SCRIPT}" ]]; then
    echo "Set WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py" >&2
    exit 2
  fi
  if [[ ! -f "${WB2_REGRID_SCRIPT}" ]]; then
    echo "WB2_REGRID_SCRIPT does not exist: ${WB2_REGRID_SCRIPT}" >&2
    exit 2
  fi
}

month_start() {
  local month="$1"
  echo "${month:0:4}-${month:4:2}-01T00"
}

month_end() {
  local month="$1"
  date -u -d "${month:0:4}-${month:4:2}-01 1 month -6 hours" +"%Y-%m-%dT%H"
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

convert_to_netcdf() {
  local input_zarr="$1"
  local output_nc="$2"

  if [[ "${CONVERT_NETCDF}" != "1" ]]; then
    return
  fi

  run_data_python "${SCRIPT_DIR}/zarr_to_netcdf.py" \
    "${input_zarr}" \
    "${output_nc}"
}

regrid_surface() {
  local input_zarr="$1"
  local output_zarr="$2"

  if [[ -e "${output_zarr}" ]]; then
    echo "Removing existing regrid output before rerun: ${output_zarr}"
    rm -rf "${output_zarr}"
  fi

  run_regrid_python "${WB2_REGRID_SCRIPT}" \
    --input_path="${input_zarr}" \
    --output_path="${output_zarr}" \
    --output_chunks="init_time=1" \
    --latitude_nodes=121 \
    --longitude_nodes=240 \
    --latitude_spacing=equiangular_with_poles \
    --regridding_method=conservative \
    --latitude_name=latitude \
    --longitude_name=longitude
}

download_month() {
  local month="$1"
  local start
  local end
  local native_zarr
  local regridded_zarr
  local regridded_nc
  local unit_args=()

  start="$(max_time "${TIME_START}" "$(month_start "${month}")")"
  end="$(min_time "${TIME_END}" "$(month_end "${month}")")"

  if [[ "${start}" > "${end}" ]]; then
    echo "Skipping ${month}; outside requested range ${TIME_START} through ${TIME_END}."
    return
  fi

  native_zarr="${SCRATCH_DIR}/aifs_native_surface_6steps_${month}.zarr"
  regridded_zarr="${SCRATCH_DIR}/aifs_surface_6steps_240x121_conservative_${month}.zarr"
  regridded_nc="${OUTPUT_DIR}/aifs_surface_6steps_240x121_conservative_${month}.nc"

  if [[ "${SKIP_EXISTING}" == "1" && "${CONVERT_NETCDF}" == "1" && -f "${regridded_nc}" ]]; then
    echo "Skipping ${month}; NetCDF already exists: ${regridded_nc}"
    return
  fi
  if [[ "${SKIP_EXISTING}" == "1" && "${CONVERT_NETCDF}" != "1" && -d "${regridded_zarr}" ]]; then
    echo "Skipping ${month}; regridded Zarr already exists: ${regridded_zarr}"
    return
  fi

  if [[ "${KEEP_SOURCE_UNITS}" == "1" ]]; then
    unit_args=(--keep-source-units)
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -d "${native_zarr}" ]]; then
    echo "Reusing ${month}; native Zarr already exists: ${native_zarr}"
  else
    run_data_python "${SCRIPT_DIR}/download_aifs_dynamical_netcdf.py" \
      "${native_zarr}" \
      --time-start "${start}" \
      --time-end "${end}" \
      --lead-hours ${LEAD_HOURS} \
      "${unit_args[@]}"
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -d "${regridded_zarr}" ]]; then
    echo "Reusing ${month}; regridded Zarr already exists: ${regridded_zarr}"
  else
    regrid_surface "${native_zarr}" "${regridded_zarr}"
  fi
  convert_to_netcdf "${regridded_zarr}" "${regridded_nc}"
}

require_regrid_script
mkdir -p "${SCRATCH_DIR}" "${OUTPUT_DIR}"

for month in ${MONTHS}; do
  download_month "${month}"
done
