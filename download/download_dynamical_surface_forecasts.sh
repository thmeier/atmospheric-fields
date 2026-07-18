#!/bin/bash
# Download native Dynamical NOAA GFS/GEFS surface forecasts to scratch, regrid
# to scratch with WeatherBench2's scripts/regrid.py, then optionally convert to
# final NetCDF in OUTPUT_DIR.
#
# Required:
#   WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py
#
# Optional:
#   SCRATCH_DIR=data/dynamical/scratch
#   OUTPUT_DIR=data/dynamical/regridded
#   MODE=all|gfs|gefs
#   TIME_START=2025-01-01T00
#   TIME_END=2025-12-31T23:59:59
#   MONTHS="202501 202502 ..."
#   LEAD_HOURS="6 12 24 48 96 192"
#   CONVERT_NETCDF=1|0
#   SKIP_EXISTING=1|0
#   KEEP_SOURCE_UNITS=1|0
#   GEFS_ENSEMBLE_MEMBERS=all|0|0,1,...
#   PYTHON=python
#   RUNNER=conda|uv

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCRATCH_DIR="${SCRATCH_DIR:-${REPO_DIR}/data/dynamical/scratch}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/data/dynamical/regridded}"
MODE="${MODE:-all}"
TIME_START="${TIME_START:-}"
TIME_END="${TIME_END:-}"
MONTHS="${MONTHS:-}"
LEAD_HOURS="${LEAD_HOURS:-6 12 24 48 96 192}"
CONVERT_NETCDF="${CONVERT_NETCDF:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
KEEP_SOURCE_UNITS="${KEEP_SOURCE_UNITS:-0}"
GEFS_ENSEMBLE_MEMBERS="${GEFS_ENSEMBLE_MEMBERS:-all}"
PYTHON="${PYTHON:-python}"
RUNNER="${RUNNER:-conda}"
WB2_REGRID_SCRIPT="${WB2_REGRID_SCRIPT:-}"
TIME_TAG="${TIME_TAG:-}"
if [[ -z "${TIME_TAG}" ]]; then
  if [[ -n "${TIME_START}" ]]; then
    TIME_TAG="${TIME_START:0:4}"
  else
    TIME_TAG="full"
  fi
fi

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

  if [[ -e "${output_zarr}" ]]; then
    echo "Removing existing regrid output before rerun: ${output_zarr}"
    rm -rf "${output_zarr}"
  fi

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

month_start() {
  local month="$1"
  echo "${month:0:4}-${month:4:2}-01T00"
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
    return
  fi

  if [[ -n "${TIME_START}" && -n "${TIME_END}" ]]; then
    generate_months
    return
  fi

  echo "${TIME_TAG}"
}

download_gfs_range() {
  local tag="$1"
  local start="$2"
  local end="$3"
  local native_zarr="${SCRATCH_DIR}/gfs_native_surface_6steps_${tag}.zarr"
  local regridded_zarr="${SCRATCH_DIR}/gfs_surface_6steps_240x121_conservative_${tag}.zarr"
  local regridded_nc="${OUTPUT_DIR}/gfs_surface_6steps_240x121_conservative_${tag}.nc"
  local time_args=()
  local unit_args=()

  if [[ "${SKIP_EXISTING}" == "1" && "${CONVERT_NETCDF}" == "1" && -f "${regridded_nc}" ]]; then
    echo "Skipping GFS ${tag}; NetCDF already exists: ${regridded_nc}"
    return
  fi
  if [[ "${SKIP_EXISTING}" == "1" && "${CONVERT_NETCDF}" != "1" && -d "${regridded_zarr}" ]]; then
    echo "Skipping GFS ${tag}; regridded Zarr already exists: ${regridded_zarr}"
    return
  fi

  if [[ -n "${start}" ]]; then
    time_args+=(--time-start "${start}")
  fi
  if [[ -n "${end}" ]]; then
    time_args+=(--time-end "${end}")
  fi
  if [[ "${KEEP_SOURCE_UNITS}" == "1" ]]; then
    unit_args=(--keep-source-units)
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -d "${native_zarr}" ]]; then
    echo "Reusing GFS ${tag}; native Zarr already exists: ${native_zarr}"
  else
    run_data_python "${SCRIPT_DIR}/download_gfs_dynamical_netcdf.py" \
      "${native_zarr}" \
      "${time_args[@]}" \
      --lead-hours ${LEAD_HOURS} \
      "${unit_args[@]}"
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -d "${regridded_zarr}" ]]; then
    echo "Reusing GFS ${tag}; regridded Zarr already exists: ${regridded_zarr}"
  else
    regrid_surface "${native_zarr}" "${regridded_zarr}" "init_time=1"
  fi
  convert_to_netcdf "${regridded_zarr}" "${regridded_nc}"
}

download_gefs_range() {
  local tag="$1"
  local start="$2"
  local end="$3"
  local native_zarr="${SCRATCH_DIR}/gefs_native_surface_6steps_${tag}.zarr"
  local regridded_zarr="${SCRATCH_DIR}/gefs_surface_6steps_240x121_conservative_${tag}.zarr"
  local regridded_nc="${OUTPUT_DIR}/gefs_surface_6steps_240x121_conservative_${tag}.nc"
  local time_args=()
  local member_args=()
  local unit_args=()

  if [[ "${SKIP_EXISTING}" == "1" && "${CONVERT_NETCDF}" == "1" && -f "${regridded_nc}" ]]; then
    echo "Skipping GEFS ${tag}; NetCDF already exists: ${regridded_nc}"
    return
  fi
  if [[ "${SKIP_EXISTING}" == "1" && "${CONVERT_NETCDF}" != "1" && -d "${regridded_zarr}" ]]; then
    echo "Skipping GEFS ${tag}; regridded Zarr already exists: ${regridded_zarr}"
    return
  fi

  if [[ -n "${start}" ]]; then
    time_args+=(--time-start "${start}")
  fi
  if [[ -n "${end}" ]]; then
    time_args+=(--time-end "${end}")
  fi
  if [[ "${GEFS_ENSEMBLE_MEMBERS}" != "all" ]]; then
    member_args=(--ensemble-members "${GEFS_ENSEMBLE_MEMBERS}")
  fi
  if [[ "${KEEP_SOURCE_UNITS}" == "1" ]]; then
    unit_args=(--keep-source-units)
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -d "${native_zarr}" ]]; then
    echo "Reusing GEFS ${tag}; native Zarr already exists: ${native_zarr}"
  else
    run_data_python "${SCRIPT_DIR}/download_gefs_dynamical_netcdf.py" \
      "${native_zarr}" \
      "${time_args[@]}" \
      --lead-hours ${LEAD_HOURS} \
      "${member_args[@]}" \
      "${unit_args[@]}"
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -d "${regridded_zarr}" ]]; then
    echo "Reusing GEFS ${tag}; regridded Zarr already exists: ${regridded_zarr}"
  else
    regrid_surface "${native_zarr}" "${regridded_zarr}" "init_time=1,ensemble_member=1"
  fi
  convert_to_netcdf "${regridded_zarr}" "${regridded_nc}"
}

download_model_months() {
  local model="$1"
  local month
  local start
  local end

  for month in $(resolve_months); do
    if [[ "${month}" == "${TIME_TAG}" && ( -z "${TIME_START}" || -z "${TIME_END}" ) ]]; then
      if [[ "${model}" == "gfs" ]]; then
        download_gfs_range "${month}" "${TIME_START}" "${TIME_END}"
      else
        download_gefs_range "${month}" "${TIME_START}" "${TIME_END}"
      fi
      continue
    fi

    start="$(max_time "${TIME_START}" "$(month_start "${month}")")"
    end="$(min_time "${TIME_END}" "$(month_end "${month}")")"

    if [[ "${start}" > "${end}" ]]; then
      echo "Skipping ${model^^} ${month}; outside requested range ${TIME_START} through ${TIME_END}."
      continue
    fi

    if [[ "${model}" == "gfs" ]]; then
      download_gfs_range "${month}" "${start}" "${end}"
    else
      download_gefs_range "${month}" "${start}" "${end}"
    fi
  done
}

require_regrid_script
mkdir -p "${SCRATCH_DIR}" "${OUTPUT_DIR}"

case "${MODE}" in
  all)
    download_model_months gfs
    download_model_months gefs
    ;;
  gfs)
    download_model_months gfs
    ;;
  gefs)
    download_model_months gefs
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use all, gfs, or gefs." >&2
    exit 2
    ;;
esac
