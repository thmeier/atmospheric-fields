#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/younes/eth_cluster_mnt/miniconda3/envs/pmlr/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

DATA_DIR="${DATA_DIR:-/cluster/courses/pmlr/teams/team07/data}"

WANDB_MODE="${WANDB_MODE:-offline}" "$PYTHON" train_discriminator.py \
  fake_nc_file="[$DATA_DIR/graphcast_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc,$DATA_DIR/fuxi_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc]" \
  test_fake_nc_file="$DATA_DIR/pangu_6steps_surf_1.5deg_2018-01-01_2018-12-31.nc" \
  +output_filename="weather_discriminator_squeezenet_all_fields_pangu_holdout_lightning.pth"
