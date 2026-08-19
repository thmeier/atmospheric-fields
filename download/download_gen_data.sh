#!/bin/bash
#SBATCH --job-name=download_wb2_gen
#SBATCH --time=08:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
# NB: this cluster rejects --cpus-per-task ("Specifying TRES per task is not allowed").
#SBATCH --mem=32G
#SBATCH --output=wb2_gen_%j.out
#SBATCH --error=wb2_gen_%j.err

# Recipe launcher for WeatherBench2 data already available in desired grids.
# No regridding is performed here.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sbatch copies the submitted script into a spool dir (/var/spool/slurm/d/jobNNN),
# so BASH_SOURCE no longer sits next to the sibling scripts this one calls.
# Fall back to the repo checkout; override with DOWNLOAD_DIR if it lives elsewhere.
if [[ ! -f "${SCRIPT_DIR}/download_era5_args.sh" ]]; then
  SCRIPT_DIR="${DOWNLOAD_DIR:-${HOME}/atmospheric-fields/download}"
fi
if [[ ! -f "${SCRIPT_DIR}/download_era5_args.sh" ]]; then
  echo "Cannot locate download_era5_args.sh (looked in ${SCRIPT_DIR}). Set DOWNLOAD_DIR." >&2
  exit 2
fi

# Remember what the caller actually set, so per-recipe defaults below can differ
# from the generic ones without silently overriding an explicit request.
CALLER_TIME_START="${TIME_START:-}"
CALLER_TIME_END="${TIME_END:-}"
CALLER_OUTPUT_DIR="${OUTPUT_DIR:-}"

TIME_START="${TIME_START:-2020-01-01}"
TIME_END="${TIME_END:-2020-12-31}"
OUTPUT_DIR="${OUTPUT_DIR:-/cluster/courses/pmlr/teams/team07/data}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pmlr}"
PYTHON="${PYTHON:-python}"

SURFACE_VARIABLES="${SURFACE_VARIABLES:-2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind mean_sea_level_pressure}"
PRESSURE_VARIABLES="${PRESSURE_VARIABLES:-temperature u_component_of_wind v_component_of_wind specific_humidity}"

# Full input/output variable set for U-Cast (Cachay et al. 2026, Table 3):
# 6 atmospheric vars x 13 WB2 pressure levels + 5 surface = 83 features, plus the
# 2 static fields. The 4 "clock" forcings are derived at runtime, not stored.
# No LEVEL filter is passed: the 240x121 store already carries exactly the 13
# WB2 levels (50 100 150 200 250 300 400 500 600 700 850 925 1000 hPa).
UCAST_VARIABLES="${UCAST_VARIABLES:-geopotential specific_humidity temperature u_component_of_wind v_component_of_wind vertical_velocity 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind mean_sea_level_pressure sea_surface_temperature geopotential_at_surface land_sea_mask}"

run_recipe() {
  local source_path="$1"
  local tag="$2"
  local variables="$3"
  shift 3

  env \
    SOURCE_PATH="${source_path}" \
    TIME_START="${TIME_START}" \
    TIME_END="${TIME_END}" \
    TAG="${tag}" \
    VARIABLES="${variables}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
    PYTHON="${PYTHON}" \
    "$@" \
    bash "${SCRIPT_DIR}/download_era5_args.sh"
}

case "${MODE:-sphericalcnn}" in
  era5-surface)
    run_recipe \
      "era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr" \
      "era5-gt" \
      "${SURFACE_VARIABLES}"
    ;;
  era5-wb13-surface)
    run_recipe \
      "era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr" \
      "era5-wb13" \
      "${SURFACE_VARIABLES}"
    ;;
  era5-ucast)
    # Full 83-feature ERA5 state for running U-Cast inference.
    # Default range is deliberately wider than the 2020 eval year: the first init
    # needs a 12h lookback, and 15-day rollouts from the last 2020 init need
    # verification targets into mid-January 2021.
    TIME_START="${CALLER_TIME_START:-2019-12-31}"
    TIME_END="${CALLER_TIME_END:-2021-01-16}"
    OUTPUT_DIR="${CALLER_OUTPUT_DIR:-/cluster/courses/pmlr/teams/team07/data/era5_ucast_2020}"
    OUTPUT_FORMAT="${OUTPUT_FORMAT:-zarr}"
    run_recipe \
      "era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr" \
      "era5-ucast83" \
      "${UCAST_VARIABLES}" \
      LEAD_HOURS=all \
      OUTPUT_FORMAT="${OUTPUT_FORMAT}"
    ;;
  era5-pressure850)
    run_recipe \
      "era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr" \
      "era5-pressure850" \
      "${PRESSURE_VARIABLES}" \
      LEVEL=850
    ;;
  graphcast)
    run_recipe \
      "graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr" \
      "graphcast" \
      "${SURFACE_VARIABLES}"
    ;;
  graphcast-2018)
    run_recipe \
      "graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr" \
      "graphcast" \
      "${SURFACE_VARIABLES}"
    ;;
  pangu)
    run_recipe \
      "pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr" \
      "pangu" \
      "${SURFACE_VARIABLES}"
    ;;
  era5-forecast)
    run_recipe \
      "era5-forecasts/2020-240x121_equiangular_with_poles_conservative.zarr" \
      "era5_forecast" \
      "${SURFACE_VARIABLES}"
    ;;
  hres)
    run_recipe \
      "hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr" \
      "ifs_hres" \
      "${SURFACE_VARIABLES}"
    ;;
  keisler)
    run_recipe \
      "keisler/2020-240x121_equiangular_with_poles_conservative.zarr" \
      "keisler" \
      "${SURFACE_VARIABLES}"
    ;;
  sphericalcnn)
    run_recipe \
      "sphericalcnn/2020-240x121_equiangular_with_poles.zarr" \
      "sphericalcnn" \
      "${SURFACE_VARIABLES}"
    ;;
  neuralgcm)
    run_recipe \
      "neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr" \
      "neuralgcm" \
      "${SURFACE_VARIABLES}"
    ;;
  fuxi)
    run_recipe \
      "fuxi/2020-240x121_equiangular_with_poles_conservative.zarr" \
      "fuxi" \
      "${SURFACE_VARIABLES}"
    ;;
  all-surface)
    for recipe in era5-surface graphcast pangu era5-forecast hres keisler sphericalcnn neuralgcm fuxi; do
      MODE="${recipe}" bash "${BASH_SOURCE[0]}"
    done
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use era5-surface, era5-wb13-surface, era5-ucast, era5-pressure850, graphcast, graphcast-2018, pangu, era5-forecast, hres, keisler, sphericalcnn, neuralgcm, fuxi, or all-surface." >&2
    exit 2
    ;;
esac
