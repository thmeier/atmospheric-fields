# Metric for Realism of Atmospheric Fields

Course project for ETH AI Center, Projects in Machine Learning Research (PMLR) 2026.

## Overview

This project develops a quantitative realism metric for atmospheric fields: a score that
captures how physically plausible a surface weather field is. The metric is trained on
[ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) reanalysis, treated
as the reference "real" distribution, and validated against machine-learning weather
forecasts (Pangu-Weather, GraphCast, FuXi, and others) as well as a suite of
physically-motivated synthetic corruptions.

All work uses the same four 1.5-degree surface fields: `2m_temperature`,
`10m_u_component_of_wind`, `10m_v_component_of_wind`, and `mean_sea_level_pressure`.

## Two complementary directions

The project investigates two independent approaches to the same problem. Each lives in its
own top-level directory with a dedicated README, scripts, and configuration.

### Discriminator: supervised adversarial discriminator

See [`Discriminator/`](Discriminator/).

A binary classifier is trained to separate real ERA5 fields (label 1) from fakes (label 0),
where fakes consist of machine-learning forecasts and synthetically corrupted fields. The raw
classifier logit is then used directly as a realism score.

- Backbones: ResNet18 and SqueezeNet (torchvision, ImageNet initialization, adapted for the
  weather-channel input).
- Hydra-based configuration, with CSV or Weights and Biases logging.
- Analyses: logit versus forecast lead time, logit versus corruption severity, leave-one-model-out
  k-fold for numerical-model comparisons, and poster figures.

Refer to [`Discriminator/README.md`](Discriminator/README.md) for the full pipeline,
configuration options, and poster reproduction.

### FeatureMetric: self-supervised latent-space metric

See [`FeatureMetric/`](FeatureMetric/).

Two self-supervised encoders, a Masked Autoencoder (MAE) and I-JEPA, are trained on ERA5 without
labels. Realism is then measured in the encoders' latent space rather than from a trained
classifier:

- Protocol 1 (linear probe): regress corruption severity from frozen latents and report R-squared.
- Protocol 2 (distribution distance): Frechet Distance and Maximum Mean Discrepancy between a clean
  reference latent distribution and corrupted or forecast distributions across a severity ladder.
- Temporal variants (`none`, `diff`, `concat`, `phase`) inject time-difference dynamics so the
  metric can react to forecast-specific artifacts rather than static state alone.

Refer to [`FeatureMetric/README.md`](FeatureMetric/README.md) and
[`FeatureMetric/CLAUDE.md`](FeatureMetric/CLAUDE.md) for architecture details and commands.

## Repository layout

```
.
├── Discriminator/     adversarial discriminator direction (supervised)
├── FeatureMetric/     self-supervised encoder direction (MAE and I-JEPA)
├── download/          shared data-download utilities (ERA5 and forecasts from WeatherBench2)
├── .gitignore
└── README.md
```

Generated and large artifacts (`data/`, `checkpoints/`, `results/`, `plots/`, `wandb/`, and
`*.out`) are git-ignored and not committed.

## Data

All fields are obtained from the [WeatherBench2](https://weatherbench2.readthedocs.io/) Google
Cloud buckets at 1.5-degree resolution and 6-hourly cadence. Shared download utilities are located
in [`download/`](download/):

- [`download/download_era5_netcdf.py`](download/download_era5_netcdf.py): download an ERA5 or
  forecast variable and time slice from a WeatherBench2 zarr store to NetCDF.
- [`download/download_era5_args.sh`](download/download_era5_args.sh): SLURM wrapper. Edit the
  source, time range, variables, and output path, then submit with `sbatch`.
- [`download/download_gen_data.sh`](download/download_gen_data.sh): recipes for the various
  forecast sources (ERA5, GraphCast, Pangu, FuXi, IFS HRES, and others).

On the cluster, data is downloaded once into the shared team directory
`/cluster/courses/pmlr/teams/team07/data`. Please do not re-download or overwrite files that
others are using.

The current ERA5 dataset covers 1.5-degree resolution, 2004 to 2023, for the four surface fields
listed above.

## Setup

Each team member uses their own conda environment (Python 3.12):

```bash
conda create -n pmlr python=3.12 -y && conda activate pmlr
```

Install dependencies. The FeatureMetric direction pins its requirements:

```bash
conda install --file FeatureMetric/requirements.txt
```

The Discriminator direction additionally uses PyTorch Lightning, torchvision, and Hydra. See
[`Discriminator/README.md`](Discriminator/README.md) for its specific requirements.

### Cluster access

```bash
ssh <your-eth-username>@student-cluster.inf.ethz.ch
```

Connect to the ETH VPN first if off-campus. Install a personal Miniconda in your home directory
and create the `pmlr` environment as described above.

## Getting started

| Task | Location |
|------|----------|
| Train or evaluate the discriminator | [`Discriminator/README.md`](Discriminator/README.md) |
| Train MAE or I-JEPA encoders and run the latent-space probes and distances | [`FeatureMetric/README.md`](FeatureMetric/README.md) |
| Download ERA5 or forecast data | [`download/`](download/) |
