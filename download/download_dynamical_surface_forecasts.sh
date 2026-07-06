#!/bin/bash
# Download native Dynamical NOAA GFS/GEFS surface forecasts to scratch, then
# regrid with WeatherBench2's scripts/regrid.py for reproducible provenance.
#
# Required:
#   WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py
#
# Optional:
#   SCRATCH_DIR=data/dynamical/scratch
#   OUTPUT_DIR=data/dynamical/regridded
#   MODE=all|gfs|gefs
#   CONVERT_NETCDF=1|0
#   GEFS_ENSEMBLE_MEMBERS=all|0|0,1,...
#   PYTHON=python
#   RUNNER=conda|uv

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCRATCH_DIR="${SCRATCH_DIR:-${REPO_DIR}/data/dynamical/scratch}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/data/dynamical/regridded}"
MODE="${MODE:-all}"
CONVERT_NETCDF="${CONVERT_NETCDF:-1}"
GEFS_ENSEMBLE_MEMBERS="${GEFS_ENSEMBLE_MEMBERS:-all}"
PYTHON="${PYTHON:-python}"
RUNNER="${RUNNER:-conda}"
WB2_REGRID_SCRIPT="${WB2_REGRID_SCRIPT:-}"

LEAD_HOURS=(6 12 24 48 96 192)

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
  if [[ "${RUNNER}" == "uv" ]]; then
    uv run "${UV_REGRID_DEPS[@]}" python "$@"
  elif [[ "${RUNNER}" == "conda" ]]; then
    "${PYTHON}" "$@"
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
  local output_chunks="$3"

  run_regrid_python "${WB2_REGRID_SCRIPT}" \
    --input_path="${input_zarr}" \
    --output_path="${output_zarr}" \
    --output_chunks="${output_chunks}" \
    --latitude_nodes=121 \
    --longitude_nodes=240 \
    --latitude_spacing=equiangular_with_poles \
    --regridding_method=conservative \
    --latitude_name=latitude \
    --longitude_name=longitude
}

download_gfs() {
  local native_zarr="${SCRATCH_DIR}/gfs_native_surface_6steps_to_2023.zarr"
  local regridded_zarr="${OUTPUT_DIR}/gfs_surface_6steps_240x121_conservative_to_2023.zarr"
  local regridded_nc="${OUTPUT_DIR}/gfs_surface_6steps_240x121_conservative_to_2023.nc"

  run_data_python "${SCRIPT_DIR}/download_gfs_dynamical_netcdf.py" \
    "${native_zarr}" \
    --lead-hours "${LEAD_HOURS[@]}"

  regrid_surface "${native_zarr}" "${regridded_zarr}" "init_time=1"
  convert_to_netcdf "${regridded_zarr}" "${regridded_nc}"
}

download_gefs() {
  local native_zarr="${SCRATCH_DIR}/gefs_native_surface_6steps_to_2023.zarr"
  local regridded_zarr="${OUTPUT_DIR}/gefs_surface_6steps_240x121_conservative_to_2023.zarr"
  local regridded_nc="${OUTPUT_DIR}/gefs_surface_6steps_240x121_conservative_to_2023.nc"
  local member_args=()

  if [[ "${GEFS_ENSEMBLE_MEMBERS}" != "all" ]]; then
    member_args=(--ensemble-members "${GEFS_ENSEMBLE_MEMBERS}")
  fi

  run_data_python "${SCRIPT_DIR}/download_gefs_dynamical_netcdf.py" \
    "${native_zarr}" \
    --lead-hours "${LEAD_HOURS[@]}" \
    "${member_args[@]}"

  regrid_surface "${native_zarr}" "${regridded_zarr}" "init_time=1,ensemble_member=1"
  convert_to_netcdf "${regridded_zarr}" "${regridded_nc}"
}

require_regrid_script
mkdir -p "${SCRATCH_DIR}" "${OUTPUT_DIR}"

case "${MODE}" in
  all)
    download_gfs
    download_gefs
    ;;
  gfs)
    download_gfs
    ;;
  gefs)
    download_gefs
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use all, gfs, or gefs." >&2
    exit 2
    ;;
esac
