#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
LOGGER="${LOGGER:-csv}"

"$PYTHON" scripts/train_kfold.py logger="$LOGGER"
