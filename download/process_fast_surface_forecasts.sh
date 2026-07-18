#!/bin/bash
# Convert fast-downloaded selected GRIB2 files to native Zarr, regrid to
# scratch with WeatherBench2's scripts/regrid.py, and optionally convert to
# final NetCDF in OUTPUT_DIR.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-gfs}"
INPUT_DIR="${INPUT_DIR:-${REPO_DIR}/data/${MODEL}_fast}"
SCRATCH_DIR="${SCRATCH_DIR:-${REPO_DIR}/data/${MODEL}_fast_processed/scratch}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/data/${MODEL}_fast_processed/regridded}"
WB2_REGRID_SCRIPT="${WB2_REGRID_SCRIPT:-}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
CYCLES="${CYCLES:-00}"
LEAD_HOURS="${LEAD_HOURS:-6 12 24 48 96 192}"
MEMBERS="${MEMBERS:-0}"
CONVERT_NETCDF="${CONVERT_NETCDF:-1}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${MODEL}" != "gfs" && "${MODEL}" != "gefs" ]]; then
  echo "MODEL must be gfs or gefs." >&2
  exit 2
fi
if [[ -z "${START_DATE}" || -z "${END_DATE}" ]]; then
  echo "Set START_DATE=YYYYMMDD and END_DATE=YYYYMMDD." >&2
  exit 2
fi
if [[ -z "${WB2_REGRID_SCRIPT}" ]]; then
  echo "Set WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py" >&2
  exit 2
fi
if [[ ! -f "${WB2_REGRID_SCRIPT}" ]]; then
  echo "WB2_REGRID_SCRIPT does not exist: ${WB2_REGRID_SCRIPT}" >&2
  exit 2
fi
WB2_REPO_DIR="$(cd "$(dirname "${WB2_REGRID_SCRIPT}")/.." && pwd)"

mkdir -p "${SCRATCH_DIR}" "${OUTPUT_DIR}"

if [[ "${MODEL}" == "gfs" ]]; then
  native_zarr="${SCRATCH_DIR}/gfs_native_surface_fast_${START_DATE}_${END_DATE}.zarr"
  regridded_zarr="${SCRATCH_DIR}/gfs_surface_fast_240x121_conservative_${START_DATE}_${END_DATE}.zarr"
  regridded_nc="${OUTPUT_DIR}/gfs_surface_fast_240x121_conservative_${START_DATE}_${END_DATE}.nc"
  output_chunks="init_time=1"
  member_args=()
else
  native_zarr="${SCRATCH_DIR}/gefs_native_surface_fast_${START_DATE}_${END_DATE}.zarr"
  regridded_zarr="${SCRATCH_DIR}/gefs_surface_fast_240x121_conservative_${START_DATE}_${END_DATE}.zarr"
  regridded_nc="${OUTPUT_DIR}/gefs_surface_fast_240x121_conservative_${START_DATE}_${END_DATE}.nc"
  output_chunks="init_time=1,ensemble_member=1"
  member_args=(--members ${MEMBERS})
fi

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_args=(--overwrite)
fi

"${PYTHON}" "${SCRIPT_DIR}/fast_grib_surface_to_zarr.py" \
  "${MODEL}" \
  "${INPUT_DIR}" \
  "${native_zarr}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --cycles ${CYCLES} \
  --lead-hours ${LEAD_HOURS} \
  "${member_args[@]}" \
  "${overwrite_args[@]}"

PYTHONPATH="${WB2_REPO_DIR}:${PYTHONPATH:-}" "${PYTHON}" "${WB2_REGRID_SCRIPT}" \
  --input_path="${native_zarr}" \
  --output_path="${regridded_zarr}" \
  --output_chunks="${output_chunks}" \
  --latitude_nodes=121 \
  --longitude_nodes=240 \
  --latitude_spacing=equiangular_with_poles \
  --regridding_method=conservative \
  --latitude_name=latitude \
  --longitude_name=longitude

if [[ "${CONVERT_NETCDF}" == "1" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/zarr_to_netcdf.py" \
    "${regridded_zarr}" \
    "${regridded_nc}"
fi
