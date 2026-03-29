#!/bin/bash
#SBATCH --account=pmlr
#SBATCH --job-name=download_era5_test
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_%j.out

# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail

#
# SETTINGS (general)
#

# the cona environment created for the dependencies
CONDA_ENV_NAME=pmlr
# this is the directory where the download_era5_netcdf.py is located
WORKDIR="$HOME/atmospheric-fields/"
# team shared output directory, i.e. only download data once,
# but be careful to not overwrite data that is needed by others
OUTPUT_DIR="/cluster/courses/pmlr/teams/team07/data"

#
# SETTINGS (data)
#

# resolution of 0.25 degree
SOURCE="gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
# start time of observations, YYYY-MM-DD format
TIME_START="2020-01-01"
# end time of observations, YYYY-MM-DD format
TIME_END="2020-01-07"
# a file with this name will be created,
# the script will fail if this already exists
OUTPUT_FILENAME="test_${TIME_START}_${TIME_END}.nc"
# do not modify
OUTPUT_FILE="$OUTPUT_DIR/$OUTPUT_FILENAME"
# which variables to pick from the observations
VARIABLES=(
    2m_temperature
    10m_u_component_of_wind
    10m_v_component_of_wind
    mean_sea_level_pressure
)

#
# DISPATCH JOB
#

mkdir -p $OUTPUT_DIR
CMD="
cd ${WORKDIR}
conda activate ${CONDA_ENV_NAME}
python download_era5_netcdf.py $SOURCE ${OUTPUT_FILE} -s $TIME_START -e $TIME_END -v ${VARIABLES[@]}
"

srun bash -c "$CMD"
