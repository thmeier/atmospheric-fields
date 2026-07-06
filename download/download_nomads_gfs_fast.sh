#!/bin/bash
# Fast selected-message GFS downloads from NOMADS using .idx byte ranges.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/data/nomads_gfs_fast}"
BASE_URL="${BASE_URL:-https://noaa-gfs-bdp-pds.s3.amazonaws.com}"
DATE="${DATE:-}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
CYCLES="${CYCLES:-00 06 12 18}"
LEAD_HOURS="${LEAD_HOURS:-6 12 24 48 96 192}"
OVERWRITE="${OVERWRITE:-0}"

args=("${OUTPUT_DIR}" --base-url "${BASE_URL}" --cycles ${CYCLES} --lead-hours ${LEAD_HOURS})

if [[ -n "${DATE}" ]]; then
  args+=(--date "${DATE}")
else
  if [[ -z "${START_DATE}" || -z "${END_DATE}" ]]; then
    echo "Set DATE=YYYYMMDD or START_DATE=YYYYMMDD and END_DATE=YYYYMMDD" >&2
    exit 2
  fi
  args+=(--start-date "${START_DATE}" --end-date "${END_DATE}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

"${PYTHON}" "${SCRIPT_DIR}/download_nomads_gfs_fast.py" "${args[@]}"
