#!/bin/bash
#SBATCH --job-name=nomads_gfs
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=nomads_gfs_%j.out
#SBATCH --error=nomads_gfs_%j.err

# Defaults are set for the ETH PMLR cluster account/user paths below. Override
# with sbatch exports only when needed:
#   DATE=YYYYMMDD
#   START_DATE=YYYYMMDD
#   END_DATE=YYYYMMDD
#   CYCLES="00 06 12 18"
#   LEAD_HOURS="6 12 24 48 96 192"
#   BASE_URL=https://noaa-gfs-bdp-pds.s3.amazonaws.com
#   OUTPUT_DIR=/work/scratch/yelberkennou/nomads_gfs_fast
#   CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV_NAME=pmlr

set -eo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pmlr}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/scratch/yelberkennou/nomads_gfs_fast}"
BASE_URL="${BASE_URL:-https://noaa-gfs-bdp-pds.s3.amazonaws.com}"
DATE="${DATE:-}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
CYCLES="${CYCLES:-00 06 12 18}"
LEAD_HOURS="${LEAD_HOURS:-6 12 24 48 96 192}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "CONDA_SH does not exist: ${CONDA_SH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

cd "${REPO_DIR}"

OUTPUT_DIR="${OUTPUT_DIR}" \
BASE_URL="${BASE_URL}" \
DATE="${DATE}" \
START_DATE="${START_DATE}" \
END_DATE="${END_DATE}" \
CYCLES="${CYCLES}" \
LEAD_HOURS="${LEAD_HOURS}" \
OVERWRITE="${OVERWRITE}" \
PYTHON=python \
bash download/download_nomads_gfs_fast.sh
