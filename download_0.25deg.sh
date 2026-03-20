#!/bin/bash
#SBATCH --job-name=download_era5_test
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_%j.out

#VENV_PATH=...
CONDA_ENV_NAME=pmlr
WORKDIR="/work/scratch/$USER"

SOURCE="gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

OUTPUT_DIR="/cluster/courses/pmlr/teams/team07/data"
mkdir -p $OUTPUT_DIR

TIME_START="2020-01-01"
TIME_END="2020-01-07"

VARIABLES=(
    2m_temperature
    10m_u_component_of_wind
    10m_v_component_of_wind
    mean_sea_level_pressure
)

#    source ${VENV_PATH} && \
CMD="cd ${WORKDIR} && \
    conda activate ${CONDA_ENV_NAME} && \
    python download_era5_netcdf.py \
        $SOURCE \
        ${OUTPUT_DIR}/era5_week.nc \
        -s $TIME_START \
        -e $TIME_END \
        -v ${VARIABLES[@]}"

srun bash -c "$CMD"
