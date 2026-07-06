# Regridding Dynamical Forecast Downloads

NOAA GFS and GEFS data from Dynamical Catalog should be regridded to match the
WeatherBench2 files already used by this project before conversion to the final
NetCDF training/evaluation format.

## Target Grid

Use the WeatherBench2 1.5-degree global grid:

| Parameter | Value |
| --- | --- |
| `latitude_nodes` | `121` |
| `longitude_nodes` | `240` |
| `latitude_spacing` | `equiangular_with_poles` |
| `regridding_method` | `conservative` |
| `latitude_name` | `latitude` |
| `longitude_name` | `longitude` |
| intermediate format | Zarr |
| final local format | NetCDF, if required by downstream repo code |

## Justification

WeatherBench2 documents that its paper/evaluation setup used `240x121`
1.5-degree files, and that all datasets were regridded with first-order
conservative regridding, using weights proportional to source/target grid-cell
area overlap:

https://weatherbench2.readthedocs.io/en/latest/data-guide.html#a-note-on-resolutions

WeatherBench2's `regrid.py` command exposes the parameters above. Its
documentation states that it supports rectilinear datasets with one-dimensional
latitude/longitude coordinates, and lists `conservative`,
`equiangular_with_poles`, `latitude_nodes`, and `longitude_nodes` as the command
options to use:

https://weatherbench2.readthedocs.io/en/latest/command-line-scripts.html#regrid

Small metadata samples from Dynamical Catalog confirm that both
`noaa-gfs-forecast` and `noaa-gefs-forecast-35-day` expose one-dimensional
`latitude` and `longitude` coordinates, satisfying the documented WeatherBench2
input-grid requirement.

## Command Template

```bash
python /path/to/weatherbench2/scripts/regrid.py \
  --input_path=INPUT.zarr \
  --output_path=OUTPUT_240x121.zarr \
  --output_chunks="init_time=1" \
  --latitude_nodes=121 \
  --longitude_nodes=240 \
  --latitude_spacing=equiangular_with_poles \
  --regridding_method=conservative \
  --latitude_name=latitude \
  --longitude_name=longitude
```

## End-to-End Pipeline

The intended workflow is:

1. Select the required Dynamical Catalog surface variables, init-time range,
   lead times, and ensemble members into a native-resolution scratch Zarr file.
2. Regrid that Zarr file with WeatherBench2.
3. Convert the regridded Zarr to NetCDF if downstream code expects local NetCDF
   inputs.

### GFS

```bash
uv run \
  --with dynamical-catalog \
  --with xarray \
  --with zarr \
  --with dask \
  --with netcdf4 \
  python download/download_gfs_dynamical_netcdf.py \
    data/gfs_native_to_2023.zarr

python /path/to/weatherbench2/scripts/regrid.py \
  --input_path=data/gfs_native_to_2023.zarr \
  --output_path=data/gfs_6steps_240x121_conservative_to_2023.zarr \
  --output_chunks="init_time=1" \
  --latitude_nodes=121 \
  --longitude_nodes=240 \
  --latitude_spacing=equiangular_with_poles \
  --regridding_method=conservative \
  --latitude_name=latitude \
  --longitude_name=longitude

uv run \
  --with xarray \
  --with zarr \
  --with dask \
  --with netcdf4 \
  python download/zarr_to_netcdf.py \
    data/gfs_6steps_240x121_conservative_to_2023.zarr \
    data/gfs_6steps_240x121_conservative_to_2023.nc
```

### GEFS

GEFS uses the Dynamical dataset id `noaa-gefs-forecast-35-day`. The downloader
defaults to all ensemble members. Use `--ensemble-members 0` for only the
control member or a smoke test.

```bash
uv run \
  --with dynamical-catalog \
  --with xarray \
  --with zarr \
  --with dask \
  --with netcdf4 \
  python download/download_gefs_dynamical_netcdf.py \
    data/gefs_native_to_2023.zarr

python /path/to/weatherbench2/scripts/regrid.py \
  --input_path=data/gefs_native_to_2023.zarr \
  --output_path=data/gefs_6steps_240x121_conservative_to_2023.zarr \
  --output_chunks="init_time=1,ensemble_member=1" \
  --latitude_nodes=121 \
  --longitude_nodes=240 \
  --latitude_spacing=equiangular_with_poles \
  --regridding_method=conservative \
  --latitude_name=latitude \
  --longitude_name=longitude

uv run \
  --with xarray \
  --with zarr \
  --with dask \
  --with netcdf4 \
  python download/zarr_to_netcdf.py \
    data/gefs_6steps_240x121_conservative_to_2023.zarr \
    data/gefs_6steps_240x121_conservative_to_2023.nc
```

## Shell Wrapper

The repository wrapper runs the wanted surface-only downloads with an explicit
scratch-staged workflow:

1. Download native-resolution Dynamical Zarr into `SCRATCH_DIR`.
2. Call WeatherBench2's `scripts/regrid.py`.
3. Write regridded outputs into `OUTPUT_DIR`.

This keeps the intermediate data and the exact WeatherBench2 command visible for
reproducibility.

The wrapper defaults to the currently active Python environment
(`RUNNER=conda`, `PYTHON=python`). Install the required packages in that
environment before running:

```bash
pip install "dynamical-catalog>=0.5.0" xarray zarr dask netCDF4
pip install "git+https://github.com/google-research/weatherbench2"
```

```bash
WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py \
SCRATCH_DIR=/scratch/$USER/dynamical_native \
OUTPUT_DIR=/cluster/courses/pmlr/teams/team07/data/dynamical \
MODE=all \
download/download_dynamical_surface_forecasts.sh
```

Useful variants:

```bash
# GFS only
WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py \
MODE=gfs \
download/download_dynamical_surface_forecasts.sh

# GEFS control member only, useful for smoke tests or memory-constrained runs
WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py \
MODE=gefs \
GEFS_ENSEMBLE_MEMBERS=0 \
download/download_dynamical_surface_forecasts.sh

# Keep only the regridded Zarr output, without final NetCDF conversion
WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py \
CONVERT_NETCDF=0 \
download/download_dynamical_surface_forecasts.sh
```

To use `uv` instead of the active conda environment, set `RUNNER=uv`.

## SLURM Wrapper

For cluster runs, use the sbatch-friendly wrapper. It activates conda from an
explicit `CONDA_SH` path, then calls the staged download/regrid wrapper:

```bash
sbatch --export=ALL,\
CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh,\
CONDA_ENV_NAME=pmlr,\
WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py,\
SCRATCH_DIR=/scratch/$USER/dynamical_native,\
OUTPUT_DIR=/cluster/courses/pmlr/teams/team07/data/dynamical,\
MODE=gfs \
download/sbatch_dynamical_surface_forecasts.sh
```

GEFS control-member smoke run:

```bash
sbatch --export=ALL,\
CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh,\
CONDA_ENV_NAME=pmlr,\
WB2_REGRID_SCRIPT=/path/to/weatherbench2/scripts/regrid.py,\
SCRATCH_DIR=/scratch/$USER/dynamical_native,\
OUTPUT_DIR=/cluster/courses/pmlr/teams/team07/data/dynamical,\
MODE=gefs,\
GEFS_ENSEMBLE_MEMBERS=0 \
download/sbatch_dynamical_surface_forecasts.sh
```

## Dynamical Catalog Caveats

Small metadata samples confirmed that both Dynamical datasets have compatible
one-dimensional latitude/longitude coordinates:

- `noaa-gfs-forecast`: dimensions include `latitude: 721`, `longitude: 1440`.
- `noaa-gefs-forecast-35-day`: dimensions include `ensemble_member: 31`,
  `latitude: 721`, `longitude: 1440`.

The wanted download is surface-only. Dynamical exposes the repo's surface fields
under Dynamical-specific source names:

| Repo name | Dynamical source name |
| --- | --- |
| `2m_temperature` | `temperature_2m` |
| `10m_u_component_of_wind` | `wind_u_10m` |
| `10m_v_component_of_wind` | `wind_v_10m` |
| `mean_sea_level_pressure` | `pressure_reduced_to_mean_sea_level` |

The downloader writes these four fields by default.

GEFS can be expensive to materialize locally even for small selections because
of the ensemble dimension and remote chunk layout. Prefer running full GEFS
downloads and regridding on the cluster, and use `--ensemble-members 0` for a
small control-member test.
