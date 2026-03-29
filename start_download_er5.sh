#!/bin/bash
#SBATCH --account=pmlr
#SBATCH --job-name=download_era5_1.5deg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

# resolution of 1.5 degree
SOURCE="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
# start time of observations, YYYY-MM-DD format
TIME_START="2004-01-01"
# end time of observations, YYYY-MM-DD format
TIME_END="2023-12-31"
# a file with this name will be created,
# the script will fail if this already exists
OUTPUT_FILENAME="era5_1.5deg_${TIME_START}_${TIME_END}.nc"
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

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

cd ${WORKDIR}
srun python download_era5_netcdf.py $SOURCE ${OUTPUT_FILE} -s $TIME_START -e $TIME_END -v ${VARIABLES[@]}
