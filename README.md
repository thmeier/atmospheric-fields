# Metric for Realism of Atmospheric Fields

Course project for **ETH AI Center — Projects in Machine Learning Research (PMLR) 2026**.

**Goal:** build a quantitative *realism metric* for atmospheric fields — a score that
captures how physically plausible a surface weather field is. The metric is trained on
[ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) reanalysis (treated
as "real") and validated against ML weather forecasts (Pangu-Weather, GraphCast, FuXi, …)
and a suite of physically-motivated synthetic corruptions.

All work uses the same four 1.5° surface fields:
`2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `mean_sea_level_pressure`.

---

## Two complementary directions

The project explores two independent approaches to the same problem. Each lives in its own
top-level folder with its own README, scripts, and configs.

### 🧭 [`Discriminator/`](Discriminator/) — supervised adversarial discriminator

A binary classifier learns to separate **real ERA5 fields** (label 1) from **fakes**
(label 0): ML forecasts and synthetically corrupted fields. The raw classifier **logit**
is used directly as a realism score.

- Backbones: ResNet18 / SqueezeNet (torchvision, ImageNet-init, weather-channel input).
- Hydra-configured training; CSV or Weights & Biases logging.
- Analyses: logit vs. forecast lead time, logit vs. corruption severity, leave-one-model-out
  k-fold for numerical-model comparisons, and poster figures.

See [`Discriminator/README.md`](Discriminator/README.md) for the full pipeline, configs, and
poster reproduction.

### 🧬 [`FeatureMetric/`](FeatureMetric/) — self-supervised latent-space metric

Two self-supervised encoders — a **Masked Autoencoder (MAE)** and **I-JEPA** — are trained on
ERA5 with no labels. Realism is then measured in the encoders' latent space rather than from a
trained classifier:

- **Protocol 1 (linear probe):** regress corruption severity from frozen latents (reports R²).
- **Protocol 2 (distribution distance):** Fréchet Distance & MMD between a clean reference
  latent distribution and corrupted / forecast distributions across a severity ladder.
- **Temporal variants** (`none` / `diff` / `concat` / `phase`) inject Δt dynamics so the metric
  can react to forecast-specific artifacts, not just static state.

See [`FeatureMetric/README.md`](FeatureMetric/README.md) and
[`FeatureMetric/CLAUDE.md`](FeatureMetric/CLAUDE.md) for architecture details and commands.

---

## Repository layout

```
.
├── Discriminator/     # adversarial discriminator direction (supervised)
├── FeatureMetric/     # self-supervised encoder direction (MAE + I-JEPA)
├── download/          # shared data-download utilities (ERA5 + forecasts from WeatherBench2)
├── .gitignore
└── README.md          # you are here
```

Generated/large artifacts (`data/`, `checkpoints/`, `results/`, `plots/`, `wandb/`, `*.out`) are
git-ignored and not committed.

---

## Data

All fields are pulled from the [WeatherBench2](https://weatherbench2.readthedocs.io/) Google Cloud
buckets (1.5° resolution, 6-hourly). Shared download utilities live in [`download/`](download/):

- [`download/download_era5_netcdf.py`](download/download_era5_netcdf.py) — download an ERA5 (or
  forecast) variable/time slice from a WeatherBench2 zarr store to NetCDF.
- [`download/download_era5_args.sh`](download/download_era5_args.sh) — SLURM wrapper; edit
  `SOURCE`, time range, variables, and output path, then `sbatch` it.
- [`download/download_gen_data.sh`](download/download_gen_data.sh) — recipes for the various
  forecast sources (ERA5, GraphCast, Pangu, FuXi, IFS HRES, …).

> **On the cluster**, data is downloaded **once** into the shared team directory
> `/cluster/courses/pmlr/teams/team07/data` — please don't re-download or overwrite files others
> are using.

Current ERA5 dataset: 1.5°, 2004–2023, the four surface fields listed above.

---

## Setup

Each team member uses their own conda environment (Python 3.12):

```bash
conda create -n pmlr python=3.12 -y && conda activate pmlr
```

Install dependencies (the `FeatureMetric/` direction pins its requirements):

```bash
conda install --file FeatureMetric/requirements.txt
```

The `Discriminator/` direction additionally uses PyTorch Lightning, torchvision, and Hydra; see
[`Discriminator/README.md`](Discriminator/README.md) for its specifics.

### Cluster access

```bash
ssh <your-eth-username>@student-cluster.inf.ethz.ch
```

Connect to the ETH VPN first if off-campus. Install a personal Miniconda in your home directory
and create the `pmlr` env as above.

---

## Getting started

| I want to… | Go to |
|---|---|
| Train / evaluate the discriminator | [`Discriminator/README.md`](Discriminator/README.md) |
| Train MAE / I-JEPA encoders and run the latent-space probes & distances | [`FeatureMetric/README.md`](FeatureMetric/README.md) |
| Download ERA5 or forecast data | [`download/`](download/) |
