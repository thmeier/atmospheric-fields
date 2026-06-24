#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
LOGGER="${LOGGER:-csv}"

ARGS=(logger="$LOGGER")

if [[ -n "${EPOCHS:-}" ]]; then
  ARGS+=(epochs="$EPOCHS")
fi

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(max_samples="$MAX_SAMPLES")
fi

if [[ -n "${BATCH_SIZE:-}" ]]; then
  ARGS+=(batch_size="$BATCH_SIZE")
fi

if [[ -n "${NUM_WORKERS:-}" ]]; then
  ARGS+=(num_workers="$NUM_WORKERS")
fi

"$PYTHON" -u scripts/train_temporal_holdout.py "${ARGS[@]}" epochs=3
