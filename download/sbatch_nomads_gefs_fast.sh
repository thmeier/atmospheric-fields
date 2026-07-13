#!/bin/bash
#SBATCH --job-name=gefs_fast
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=gefs_fast_%j.out
#SBATCH --error=gefs_fast_%j.err

# Defaults are set for the ETH PMLR cluster account/user paths below. Override
# with sbatch exports only when needed:
#   START_DATE=20201001
#   END_DATE=20231231
#   CYCLES="00"
#   LEAD_HOURS="6 12 24 48 96 192"
#   MEMBERS="0"       # use "all" for control + 30 perturbed members
#   OUTPUT_DIR=/work/scratch/yelberkennou/gefs_fast
#   CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV_NAME=pmlr

set -eo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pmlr}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/scratch/yelberkennou/gefs_fast}"
BASE_URL="${BASE_URL:-https://noaa-gefs-pds.s3.amazonaws.com}"
DATE="${DATE:-}"
START_DATE="${START_DATE:-20201001}"
END_DATE="${END_DATE:-20231231}"
CYCLES="${CYCLES:-00}"
LEAD_HOURS="${LEAD_HOURS:-6 12 24 48 96 192}"
MEMBERS="${MEMBERS:-0}"
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
MEMBERS="${MEMBERS}" \
OVERWRITE="${OVERWRITE}" \
PYTHON=python \
bash download/download_nomads_gefs_fast.sh
