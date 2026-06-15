#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data}"
LOGGER="${LOGGER:-csv}"

"$PYTHON" scripts/train_discriminator.py \
  --config-name poster_config \
  logger="$LOGGER" \
  fake_nc_file="[$DATA_DIR/graphcast_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc,$DATA_DIR/fuxi_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc]" \
  test_fake_nc_file="$DATA_DIR/pangu_6steps_surf_1.5deg_2018-01-01_2018-12-31.nc" \
  +output_filename="weather_discriminator_squeezenet_all_fields_pangu_holdout_lightning.pth"
