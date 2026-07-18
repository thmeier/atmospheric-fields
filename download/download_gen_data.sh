#!/bin/bash
#SBATCH --time=04:00
#SBATCH --account=pmlr_jobs

# Recipe launcher for WeatherBench2 data already available in desired grids.
# No regridding is performed here.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TIME_START="${TIME_START:-2020-01-01}"
TIME_END="${TIME_END:-2020-12-31}"
OUTPUT_DIR="${OUTPUT_DIR:-/cluster/courses/pmlr/teams/team07/data}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pmlr}"
PYTHON="${PYTHON:-python}"

SURFACE_VARIABLES="${SURFACE_VARIABLES:-2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind mean_sea_level_pressure}"
PRESSURE_VARIABLES="${PRESSURE_VARIABLES:-temperature u_component_of_wind v_component_of_wind specific_humidity}"

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
    echo "Unsupported MODE=${MODE}. Use era5-surface, era5-wb13-surface, era5-pressure850, graphcast, graphcast-2018, pangu, era5-forecast, hres, keisler, sphericalcnn, neuralgcm, fuxi, or all-surface." >&2
    exit 2
    ;;
esac
