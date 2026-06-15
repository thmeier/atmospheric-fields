#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

"$PYTHON" scripts/evaluate_discriminator.py
