# Atmospheric Field Discriminator

This directory contains an experimental discriminator pipeline for assigning
realism scores to atmospheric fields. The main discriminator learns a binary
classification task: ERA5/reference fields are labeled real, while model
forecasts and optional synthetic corruptions are labeled fake. The raw logit is
then used to derive a realism score in the evaluation and analysis scripts.

## Directory Structure

- `conf/config.yaml`: default Hydra configuration for data paths, variables,
  train/test ranges, model choice, augmentation, logging, and outputs.
- `scripts/train_discriminator.py`: main training entrypoint. Trains one ResNet18
  or SqueezeNet discriminator and writes a `.pth` file under `output_dir`.
- `scripts/train_kfold.py`: trains leave-one-neural-model-out discriminators and
  one full-pool discriminator for numerical-model comparisons. In the k-fold
  setup, each fold holds out one neural forecast model and otherwise uses the
  available ERA5 range and all available non-held-out neural forecast files.
- `scripts/evaluate_discriminator.py`: computes test accuracy/loss and saves map panels
  for high-logit, uncertain, fooled, and obvious-fake samples.
- `scripts/plot_logits_vs_lead_time.py`: plots mean discriminator logits across forecast
  lead times for every file in `comparison_files`.
- `scripts/plot_logits_vs_lead_time_kfold.py`: lead-time plot for k-fold holdout
  discriminators.
- `scripts/plot_logits_vs_disturbance.py`: sensitivity analysis on clean ERA5 fields
  after synthetic disturbances (strength of disturbance is taken to be the
  analogue of lead time).
- `scripts/plot_logits_vs_disturbance_kfold.py`: disturbance sensitivity analysis across
  available k-fold/full-pool discriminators.
- `scripts/plot_poster_dist_severity.py`: poster figure combining discriminator
  reverse-KL-style scores with cached I-JEPA/FID data.
- `scripts/analysis_utils.py`: shared inference helpers for normalization, lead-time
  datasets, device selection, and checkpoint loading.
- `scripts/corruptions.py`: tensor corruptions used during training and some poster
  analyses.
- `run_*.sh`: small shell wrappers. They default to `python` and can be
  customized with environment variables.
- `artifacts/`: generated plots retained for reference.
- `legacy/`: older exploratory scripts kept out of the active pipeline.
- `weather-discriminator*/`: Lightning/W&B checkpoint directories from previous
  runs.
- `poster_fid_*_nontemporal.npz`: cached FID/I-JEPA arrays consumed by
  `scripts/plot_poster_dist_severity.py`.

## Logging

Training does not require W&B by default. The `logger` option controls Lightning
experiment logging:

- `logger=csv`: local CSV logs under `output_dir/lightning_logs` (default).
- `logger=wandb`: explicit Weights & Biases logging using `project_name`.
- `logger=none`: disable Lightning experiment logging.

Examples:

```bash
python scripts/train_discriminator.py logger=csv
python scripts/train_discriminator.py logger=wandb project_name=weather-discriminator
python scripts/train_discriminator.py logger=none
```

For W&B offline logging, use:

```bash
WANDB_MODE=offline python scripts/train_discriminator.py logger=wandb
```

## Common Commands

Train the default discriminator:

```bash
python scripts/train_discriminator.py
```

Use a custom data directory:

```bash
DATA_DIR=/path/to/netcdf python scripts/train_discriminator.py
python scripts/train_discriminator.py data_dir=/path/to/netcdf
```

Train a one-field discriminator:

```bash
python scripts/train_discriminator.py variables=null selected_variable=2m_temperature
```

Train with a different backbone:

```bash
python scripts/train_discriminator.py model_name=resnet18
python scripts/train_discriminator.py model_name=squeezenet
```

Train with custom fake files:

```bash
python scripts/train_discriminator.py \
  'fake_nc_file=[/path/to/pangu.nc,/path/to/fuxi.nc]' \
  train_fake_range='["2020-01-01","2020-12-31"]'
```

Train k-fold holdout discriminators:

```bash
python scripts/train_kfold.py
```

### K-Fold Experiment

The k-fold experiment is configured by `conf/kfold_config.yaml`. It trains one
discriminator per neural forecast model by holding that model out and training
on all configured non-held-out neural forecast files in `fake_nc_file`. It also
trains a full-pool discriminator for numerical-model comparisons. The current
multi-field configuration uses `temperature`, `u_component_of_wind`, and
`v_component_of_wind`, so checkpoint names use the `all_fields` tag.
Training-time augmentation uses the configured `corruption_types`; the current
model k-fold setup trains on `gaussian_blur`, `grf`, `pixel_replace`, and
`wind_patch_shuffle`, while leaving `hf_noise` and `wind_rotation` as held-out
corruption probes. These held-out corruptions do not define the k-fold split;
the k-fold split is still the forecast model holdout.

On the cluster, run:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 02:00 bash run_finetune_kfold.sh
```

For a quick smoke test:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
MAX_SAMPLES=128 EPOCHS=1 BATCH_SIZE=8 NUM_WORKERS=0 \
srun -A pmlr -t 00:10 bash run_finetune_kfold.sh
```

K-fold checkpoints are written to:

```text
results/kfold_checkpoints/
```

Generate the lead-time comparison plot:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 00:10 python scripts/plot_logits_vs_lead_time_kfold.py
```

Generate the disturbance sensitivity plots:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 00:10 python scripts/plot_logits_vs_disturbance_kfold.py
```

This evaluates all model-holdout k-fold checkpoints against every configured
training corruption plus the held-out corruption probes, labeling the held-out
probe curves in the plot title.

### Corruption K-Fold Experiment

The corruption k-fold experiment uses the same forecast/ERA5 time split as the
model k-fold experiment, but the fold axis is a configured list of synthetic
corruptions. Each entry in `corruption_kfold_holdouts` is a list of corruption
families held out together; the corresponding discriminator is trained on
`corruption_kfold_types` minus that list.

Run the grouped corruption-holdout training:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 02:00 bash run_corruption_kfold.sh
```

For a quick smoke test:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
MAX_SAMPLES=128 EPOCHS=1 BATCH_SIZE=8 NUM_WORKERS=0 \
srun -A pmlr -t 00:10 bash run_corruption_kfold.sh
```

Checkpoints are written to:

```text
results/corruption_kfold_checkpoints/
```

Plot logits versus corruption strength. This writes three views: all curves,
trained-on corruption curves only, and held-out corruption curves only.

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 00:10 python scripts/plot_logits_vs_corruption_kfold.py
```

### Standard Metric Baselines

The tracked baseline pipeline compares unpaired forecast or corrupted-field
distributions with ERA5. Its canonical recurring split uses days 1–15 of every
month for training, days 20–26 for testing, and days 16–31 for the separate ERA5
null comparison. Forecast membership is determined by valid time, and every
forecast must have an exact ERA5 match; there is no nearest-time fallback.
Forecast evaluation is restricted to the shared configured 2020 coverage.
Corruption evaluation uses ERA5 from 2004–2022 and an evenly spaced cap of 1,000
test samples by default.

The active baseline configuration is `conf/baseline_config.yaml`. It runs the
joint four-field T2M/U10/V10/MSL case by default; scalar experiments remain
available by overriding `baseline.variables` and
`target_discriminator.variables`. The forecast catalog contains GraphCast, Pangu-Weather, FuXi, IFS HRES, ERA5
Forecast, SWIFT, and one deterministic UCast member; ensemble averaging is not
used.

The default corruption suite contains Gaussian blur, high-frequency noise, GRF,
`pixel_replace`, wind patch shuffling and rotation when U/V are available, a
2x2-pixel checkerboard, zonal scanlines, `hemisphere_splice`, and
`field_splice`. The latter independently replaces each complete variable field
with probability `severity / maximum_severity`, using a distinct deranged ERA5
donor permutation for every variable. Pixel replacement spans `[0, 0.01]`,
Gaussian blur spans its native `[0, 1]` scale, and most other corruptions span
`[0, 0.2]`. Data-dependent splice critics train only against complete
replacement, while evaluation retains the complete probability curve.

The available metrics include pointwise mean and standard-deviation
discrepancies, field energy distance, linear and log zonal-spectrum L2, sliced
Wasserstein/Cramer–Wold variants, RBF MMD, global-mean Wasserstein, and SCWD.
Mean and standard-deviation discrepancies use ordinary grid-cell moments. The
joint MMD uses one ERA5-fitted median-heuristic bandwidth per field block within
a single product RBF kernel. Multi-field global-mean Wasserstein follows a joint
Vissio-style Ulam discretization, and multi-field SCWD performs exact empirical
quadratic OT in the field-response space at each anchor. The latter uses an
evenly spaced 256-sample budget by default; scalar SCWD uses 200 quantiles.

Metric extraction is streamed. Full-sample mean, variance, spectra, and global
means are accumulated without retaining all fields. Pairwise field energy and
MMD use 256 evenly spaced samples, sliced metrics retain 4,096 total spatial
coordinates and 64 projections, and multi-field SCWD uses 256 samples. CSV
outputs record both `n_samples` and `pairwise_n_samples`. Bootstrap uncertainty
is disabled on the standard curves; reported values are point estimates. The
optional bootstrap-null workflow instead resamples disjoint ERA5 partitions to estimate a
null distribution and its configured upper-quantile detection threshold. Run it
separately with:

```bash
cd /home/yelberkennou/atmospheric-fields/Discriminator
sbatch scripts/submit_bootstrap_null.sh
```

Use the tracked pipeline rather than the legacy standalone evaluation wrapper:

```bash
cd /home/yelberkennou/atmospheric-fields/Discriminator
sbatch scripts/submit_nosfno_paper_pipeline.sh
```

Every invocation writes to an immutable directory:

```text
results/baselines/monthly_valid_time_2020/surface/pipeline_runs/<pipeline-id>/
├── resolved_config.yaml
├── manifest.json
└── <variable-tag>/
    ├── data/
    ├── models/target_discriminators/
    └── plots/
        └── paper/
```

Every figure is accompanied by an NPZ data bundle and a title-less variant.
Dashboard mode saves PNG/NPZ by default; paper mode additionally saves PDF,
uses manuscript-scale typography, and places outputs under `plots/paper/`.
Evaluation CSV/NetCDF artifacts and PNG/PDF/NPZ plot bundles are uploaded to W&B
when the corresponding pipeline upload switches are enabled.

### Learned-Metric Blind-Spot Search

The standalone adversarial-corruption experiment searches for coherent
temperature perturbations that remain within the clean sampling variability of
SCWD and zonal log-spectrum distance while being easy for a learned metric to
detect. It trains a dependency-free coarse residual U-Net cooperatively with
the usual SqueezeNet discriminator at fixed normalized RMS values of 0.05,
0.10, and 0.20. The residual has zero area-weighted mean, is generated at
quarter resolution, and is penalized for excessive peaks or spatially
concentrated energy. Together with random longitude rolls, this strongly
suppresses single-pixel and fixed-corner solutions.

Training uses only `train_real_range`; normalization and clean metric thresholds
are fitted there. After freezing each generator, the script trains a fresh
SqueezeNet and evaluates it on `test_real_ranges`. Final SCWD uses all 60x120
anchors and the standard test-fitted ERA5 normalization. A matched random coarse
perturbation is reported as a control.

Run the complete RMS sweep:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 2-00:00 \
  python scripts/train_adversarial_corruption.py \
  adversarial_corruption.scratch_dir=/tmp
```

For a quick pipeline check:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 00:30 \
  python scripts/train_adversarial_corruption.py \
  'adversarial_corruption.rms_values=[0.10]' \
  adversarial_corruption.joint_epochs=1 \
  adversarial_corruption.fresh_discriminator_epochs=1 \
  adversarial_corruption.max_train_samples=64 \
  adversarial_corruption.max_test_samples=64 \
  adversarial_corruption.null_calibration_pairs=2 \
  adversarial_corruption.training_scwd_anchors=32 \
  adversarial_corruption.training_scwd_anchor_banks=2 \
  adversarial_corruption.pretrained_discriminator=false \
  baseline.scwd_anchor_lat_points=4 \
  baseline.scwd_anchor_lon_points=8 \
  baseline.scwd_domain_lat_points=9 \
  baseline.scwd_domain_lon_points=16
```

Outputs are namespaced under
`results/adversarial_corruption/2m_temperature/`, with generator and
discriminator checkpoints, training histories, held-out summaries, NetCDF/PNG
case studies, and a frontier plot across RMS budgets.

### Finite-SCWD Null-Space Case Study

The standalone null-space diagnostic projects a deterministic pixel-checkerboard
field into the null space of the exact finite 60x120 Wendland response matrix.
It applies that direction to temporal-holdout ERA5 at normalized RMS values
0.05, 0.10, and 0.20 and compares the result with the identical clean sample
distribution. This demonstrates a blind spot of the sampled implementation;
it is not a claim that the perturbation lies in the null space of continuous
SCWD. A matched, unprojected pattern is included as a nonzero control.

Run the 128-sample case study, visualized on August 23, 2018:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 00:15 \
  python scripts/plot_scwd_null_space_case_study.py
```

The figure, NetCDF fields, metrics CSV, and resolved configuration are written
under `results/baselines/temporal_holdout/surface/2m_temperature/scwd_null_space/`.
The same run also writes `near_null_gallery.png`, `near_null_gallery.nc`, and
`near_null_metrics.csv`. These compare equally energetic equatorial checker
texture, meridional scanlines, a 2x2-pixel checkerboard, and zonal scanlines.
Set `scwd_null_space.near_null_gallery_rms` to change their shared RMS, or
override `scwd_null_space.near_null_patterns` to select a subset.

Finally, the run creates `hemisphere_splice.png`, `hemisphere_splice.nc`, and
`hemisphere_splice_metrics.csv`. For metric evaluation, the southern halves are
a seeded derangement of the same ERA5 sample set: filters contained within one
hemisphere retain the original marginal distribution, while only filters near
the hard equatorial seam can observe the broken north-south dependence. The
figure combines the August 23 northern state with a reproducibly random southern
donor and shows both sources, the splice, and its signed difference. Configure
this with `hemisphere_splice_seed`, `hemisphere_splice_latitude`, or
`hemisphere_splice_enabled` under `scwd_null_space`.

### Temporal Holdout Experiment

The temporal experiment uses the same recurring monthly protocol as the
baselines: forecast valid times on days 1–15 train the discriminator and valid
times on days 20–26 evaluate it. Every fake forecast is paired with ERA5 at the
exact same valid timestamp; its matched ERA5 sample supplies the real class.
Each model uses all of its configured files (2018 and/or 2020), so train and test
come from the same seasonal and model-availability distribution. With the
default multi-field config, checkpoint names use the `all_fields` tag.

On the cluster, run:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 02:00 bash run_temporal_holdout.sh
```

For a quick smoke test:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
MAX_SAMPLES=128 EPOCHS=1 BATCH_SIZE=8 NUM_WORKERS=0 \
srun -A pmlr -t 00:10 bash run_temporal_holdout.sh
```

Temporal holdout checkpoints are written to:

```text
results/temporal_holdout_checkpoints/
```

Generate the lead-time comparison plot:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 00:10 python scripts/plot_logits_vs_lead_time_temporal_holdout.py
```

Create an interpretable held-out forecast case study at 24 h. The script ranks
test forecasts by their scalar realism logit, selects low/median/high examples,
and plots forecast, matching valid-time ERA5, and the pre-pooling spatial logit
map. The map's spatial mean is checked against the scalar logit before plotting.

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 00:30 python scripts/plot_temporal_logit_map_case_study.py
```

Use `case_study_model`, `case_study_lead_hour`, and
`case_study_logit_quantiles` in `conf/config.yaml` (or Hydra overrides) to
choose the temporal discriminator, lead, and examples. Outputs are written to
`results/temporal_logit_map_case_study/` with a CSV describing selected samples.
Set `case_study_initialization_date="2018-09-12"` to select examples only from
that UTC initialization day.

Evaluate one trained discriminator:

```bash
python scripts/evaluate_discriminator.py
```

Plot lead-time degradation:

```bash
python scripts/plot_logits_vs_lead_time.py
python scripts/plot_logits_vs_lead_time_kfold.py
python scripts/plot_logits_vs_lead_time_temporal_holdout.py
```

Plot sensitivity to synthetic disturbances:

```bash
python scripts/plot_logits_vs_disturbance.py
python scripts/plot_logits_vs_disturbance_kfold.py
```

Build the poster figure:

```bash
python scripts/plot_poster_dist_severity.py
```

Force the poster script to recompute its discriminator curves:

```bash
python scripts/plot_poster_dist_severity.py --recompute
```

## Poster Training Setup

The poster discriminator figure is based on two trained discriminators. Both use
ERA5 as the reference distribution and train against neural forecast models plus
synthetic corruptions. Numerical models such as `IFS HRES` and `ERA5 Forecast`
are not used as fake training data.

The executed setup is captured in `conf/poster_config.yaml`. Use that config for
poster reproduction rather than the general-purpose `conf/config.yaml`.

Shared split and model settings used for the poster:

- Real/reference training data: ERA5 from `2008-01-01` to `2017-12-31`, plus
  ERA5 from `2020-01-01` to `2020-12-31`.
- Real/reference test data: ERA5 up to `2007-12-31`, ERA5 from `2018-01-01` to
  `2019-12-31`, and ERA5 from `2021-01-01` onward.
- Forecast training data: neural forecast files from `2020-01-01` to
  `2020-12-31`.
- Forecast holdout/evaluation data: forecast files from `2018-01-01` to
  `2019-12-31`.
- Lead times used by the poster/evaluation setup: `6`, `12`, `24`, `48`, `96`,
  and `192` hours.
- Fields: `2m_temperature`,
  `10m_u_component_of_wind`, `10m_v_component_of_wind`, and
  `mean_sea_level_pressure`.
- Backbone: `squeezenet1_1` with ImageNet initialization. The input convolution
  is resized for the weather channels, and the final classifier projection is
  replaced with a scalar logit head trained by `BCEWithLogitsLoss`.
- Optimizer: one AdamW learning rate was used for all parameters. No separate
  backbone learning-rate multiplier was used.
- Training schedule: `epochs=10`, `batch_size=64`, `learning_rate=1e-4`,
  `precision=16-mixed`, and `num_workers=4`.
- Normalization: inputs are z-scored using ERA5/reference statistics from
  `train_real_range`.

The two poster discriminators:

1. GraphCast-holdout discriminator

   This is the primary poster model loaded by
   `scripts/plot_poster_dist_severity.py` when using `conf/poster_config.yaml`:

   ```text
   results/weather_discriminator_squeezenet_all_fields_lightning.pth
   ```

   It was trained with GraphCast held out. Its fake training pool contained the
   other neural forecast models, Pangu and FuXi, using their 2020 forecast
   files. It is used for:
   - the GraphCast forecast curve;
   - the IFS HRES/numerical forecast curve;
   - the held-out corruption curves.

   Example:

   ```bash
   python scripts/train_discriminator.py \
     --config-name poster_config \
     logger=csv \
     fake_nc_file="[$DATA_DIR/pangu_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc,$DATA_DIR/fuxi_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc]" \
     test_fake_nc_file="$DATA_DIR/graphcast_6steps_surf_1.5deg_2018-01-01_2018-12-31.nc"
   ```

2. Pangu-holdout discriminator

   This model was trained with Pangu held out. Its fake training pool contained
   the other neural forecast models, GraphCast and FuXi, using their 2020
   forecast files. It is used by the poster script only for the
   Pangu-Weather forecast curve.

   Expected output:

   ```text
   results/weather_discriminator_squeezenet_all_fields_pangu_holdout_lightning.pth
   ```

   Example:

   ```bash
   ./run_train_pangu_holdout.sh
   ```

   Equivalent direct command:

   ```bash
   python scripts/train_discriminator.py \
     --config-name poster_config \
     logger=csv \
     fake_nc_file="[$DATA_DIR/graphcast_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc,$DATA_DIR/fuxi_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc]" \
     test_fake_nc_file="$DATA_DIR/pangu_6steps_surf_1.5deg_2018-01-01_2018-12-31.nc" \
     +output_filename=weather_discriminator_squeezenet_all_fields_pangu_holdout_lightning.pth
   ```

## K-Fold Setup

The k-fold runs use a different split principle from the poster train/test-time
split above. Each leave-one-neural-model-out discriminator is trained with the
held-out forecast model removed from the fake training pool; the other neural
forecast model files are used. Their filename date spans are merged into one
model-time union. Forecast fake samples are drawn from that union, while
ERA5/reference real samples are drawn from the complement of that union within
the available ERA5 file span.

The full-pool k-fold discriminator keeps all neural forecast model files in the
fake training pool and is used for numerical-model comparisons.

For plotting, `test_real_ranges` and `test_fake_range` use the model-time union.
This means the discriminator sees ERA5 real samples only outside the forecast
evaluation periods during training, then scores ERA5 and forecast samples on the
same model-time union at test time.

Training sample composition:

- Labels: ERA5/reference samples are `1.0`; forecasts and corrupted samples are
  `0.0`.
- Epoch composition in balanced mode: half ERA5/reference samples and half fake
  samples.
- The balanced dataset length is `2 * max(n_real_times, n_forecast_time_leads)`.
  Real and forecast indices are taken modulo their available sample counts, so
  the smaller side is reused within an epoch. The PyTorch `DataLoader` then
  shuffles these balanced indices each epoch.
- Fake composition when `augment=true`: the fake half is sampled uniformly from
  forecast, corrupted ERA5, and corrupted forecast categories.
- Fake category selection is random at sample time. Repeated visits to the same
  balanced index can therefore yield different fake categories and corruptions
  across epochs.
- Forecast consistency: a forecast sample is one coherent NetCDF slice at one
  initialization time and one lead time; all configured fields come from that
  same forecast sample.
- Forecast lead-time samples are enumerated over all configured lead times for
  each selected initialization time, then reused by modulo indexing if needed for
  balancing.

Training corruptions and poster corruption holdouts:

- During training, corruption type is selected first from `corruption_types`.
  The type is sampled uniformly from the configured list.
- With the poster training corruption pool, training used `blur`, `grf`,
  `pixel_replace`, and `wind_patch_shuffle`.
- If the chosen corruption is fieldwise (`blur`, `grf`, `pixel_replace`, or
  `hf_noise`), fields are then selected independently with probability
  `field_corruption_prob`, with at least one field forced. A separate severity is
  sampled for each selected field.
- Corruption severities are random and sampled from
  `corruption_severity_max * U(0, 1) ** corruption_severity_power`, which biases
  toward weaker corruptions when `corruption_severity_power > 1`.
- `wind_patch_shuffle` and `wind_rotation` are not fieldwise. They act jointly
  on the U/V wind channels and leave temperature/pressure untouched.
- The poster corruption probes are `hf_noise` and `wind_rotation`. Under the
  poster training corruption pool, these were held out from training.

Poster plotting and cache details:

- `scripts/plot_poster_dist_severity.py` loads `conf/poster_config.yaml` by
  default. Use `--config-name config` only for non-poster experiments.
- The poster config includes a `poster:` provenance block with the two model
  paths, the two fake training pools, and the held-out forecast/corruption
  labels.
- `scripts/plot_poster_dist_severity.py` computes discriminator curves once and
  caches them in `results/poster_discriminator_reverse_kl_data.npz` unless
  `--recompute` is passed.
- Poster curve evaluation is deterministic for a fixed `--seed`. If
  `--n-samples` is smaller than the available time count, samples are selected
  by evenly spaced indices over time, not by random sampling with replacement.
  The poster command default is `--n-samples 400`.
- The GraphCast-holdout model path from the poster config is used for the
  held-out corruption curves, GraphCast, and IFS HRES. The Pangu-holdout model
  path from the poster config is loaded only for the Pangu-Weather curve.
- The script also needs cached I-JEPA/FID inputs:
  `poster_fid_severity_data_nontemporal.npz` and
  `poster_fid_leadtime_data_nontemporal.npz`. It does not compute those files.

## Important Options

- `data_dir`: base directory for NetCDF files. Defaults to `DATA_DIR` from the
  environment, or `data` if that variable is unset.
- `real_nc_file`, `fake_nc_file`: training reference and forecast files.
- `test_real_nc_file`, `test_fake_nc_file`: evaluation reference and forecast
  files.
- `variables`: list of input channels. Set `variables=null` to use only
  `selected_variable`.
- `selected_variable`: field used by single-variable analyses and output names.
- `monthly_split.train_days`, `monthly_split.test_days`: canonical recurring calendar split.
- `monthly_split.null_comparison_days`: ERA5 null-divergence comparison interval.
- `monthly_split.corruption_time_range`: complete ERA5 years used by corruption experiments.
- `monthly_split.model_valid_time_ranges`: allowed forecast valid-time coverage.
- Legacy `train_*` and `test_*` ranges remain for poster and k-fold configs.
- `lead_times`: forecast lead hours to use from forecast files.
- `model_name`: `squeezenet` or `resnet18`.
- `augment`: if true, fake samples include forecast fields, corrupted ERA5, and
  corrupted forecasts.
- `corruption_types`: training-time corruption pool. Supported names include
  `blur`, `grf`, `hf_noise`, `pixel_replace`, `wind_patch_shuffle`, and
  `wind_rotation`.
- `corruption_severity_max`: maximum training-time corruption severity.
- `corruption_severity_power`: samples lower severities more often when greater
  than `1`.
- `field_corruption_prob`: probability each field is selected for fieldwise
  corruptions.
- `batch_size`, `epochs`, `learning_rate`, `num_workers`, `precision`: training
  performance and optimization controls.
- `output_dir`: target directory for `.pth` weights, plots, CSV logs, and cached
  analysis outputs.
- `comparison_files`: named forecast/numerical files used by comparison plots and
  k-fold training.

## Outputs

## Target-specific discriminator baselines

### Tracked baseline pipeline

`scripts/run_baseline_pipeline.py` is the canonical entry point. It executes
selected stages in dependency order within one isolated pipeline-run directory:

1. optional dataset-level histogram-map fitting;
2. target-specific discriminator training and held-out testing;
3. standard metric evaluation;
4. discriminator metric evaluation; and
5. artifact-only plotting.

Each stage gets a clearly named W&B run under the shared pipeline ID. Training
uses one run per architecture and target; evaluation tables, checkpoints,
resolved configuration, plot bundles, and the final stage manifest are uploaded
as versioned artifacts according to `pipeline.wandb`. Omitting a producing stage
does not implicitly fetch its inputs: provide `pipeline.input_checkpoint_dir`
and, when applicable, `pipeline.input_histogram_matching_dir` explicitly.

The normal complete pipeline, including optional SFNO probes, can be submitted
with `scripts/submit_baseline_pipeline.sh`. The tested paper-oriented run without
SFNO is:

```bash
cd /home/yelberkennou/atmospheric-fields/Discriminator
sbatch scripts/submit_nosfno_paper_pipeline.sh
```

This wrapper assigns `nosfno-paper-<job-id>` to both the local run directory and
W&B group, disables both SFNO configuration paths, runs training, both metric
evaluations, and plotting, and opts into manuscript-sized PNG/PDF/NPZ bundles.
The cluster account supplies one GPU, two CPUs, and 24 GB without explicit TRES
requests. Hydra overrides may be appended after the script name.

A small end-to-end check is available as:

```bash
sbatch scripts/submit_nosfno_pipeline_smoke_test.sh
```

It uses one epoch, 32 train/evaluation samples, Gaussian blur and field splice,
and the two inexpensive moment metrics, while still traversing every pipeline
stage and exercising online W&B uploads and paper bundle generation.

For interactive or selective execution:

```bash
conda activate pmlr
export DATA_DIR=/cluster/courses/pmlr/teams/team07/data

python scripts/run_baseline_pipeline.py \
  "pipeline.stages=[train_discriminators,evaluate_standard_metrics,evaluate_discriminator_metrics,plot]" \
  target_discriminator.sfno.enabled=false \
  baseline.discriminator.sfno.enabled=false

# Reuse an explicitly selected checkpoint directory for evaluation.
python scripts/run_baseline_pipeline.py \
  "pipeline.stages=[evaluate_discriminator_metrics,plot]" \
  pipeline.input_checkpoint_dir=/absolute/path/to/models/target_discriminators
```

Histogram matching is disabled by default. When enabled, it fits a training-only,
dataset-level scalar quantile map for each variable and target, pooling values
across times and grid points, then applies that frozen pointwise mapping before
standardization throughout training and evaluation. Run the fitting stage in the
same invocation, or provide a previously fitted map directory explicitly.

Target discriminator training writes train/test loss and accuracy, normalized
held-out logit histograms for every lead or severity, integrated-gradient
interpretability galleries, and checkpoints. SqueezeNet is the default raw-field
critic. The optional attention variant adds a globally mixed token. The optional
SFNO path feeds all four raw surface fields through the pretrained encoder and
trains either a linear or residual-MLP probe; it can instead insert ERA5 context
for non-target channels using
`target_discriminator.sfno.use_era5_context_for_non_target_fields=true`.

Corruption critics draw uniformly from the same discrete nonzero severity grid
used at test time by default. `hemisphere_splice` and `field_splice` train only at
complete replacement so unchanged samples are not labeled fake. Donor
permutations are refreshed each training epoch and are independently deranged by
field for `field_splice`.

The pipeline manifest and resolved configuration live directly under the
pipeline-run directory. Evaluation-only and plotting-only invocations create
new output directories, preserving prior runs, while reading checkpoints only
from the explicitly supplied input directory.

An optional `squeezenet_attention` target discriminator adds global mixing to
the temperature-only SqueezeNet baseline. Its final 512x7x14 (or padded
512x7x15) feature map is projected to 64 channels, augmented with fixed
latitude and periodic-longitude Fourier positions, and processed with one
four-head Transformer layer (128-wide GELU feed-forward block, dropout 0.1).
A learned global token produces the scalar logit. This adds 66,048 parameters
to the 721,857-parameter baseline. Because the logit comes from a globally mixed
token, it has no exact additive pre-pooling logit-map decomposition.

Train only this architecture, retaining existing SqueezeNet and SFNO probe
checkpoints:

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
  srun -A pmlr -t 2-00:00 \
    python scripts/train_target_discriminator_baselines.py \
      target_discriminator.train_squeezenet=false \
      target_discriminator.train_attention_squeezenet=true \
      target_discriminator.sfno.enabled=false
```

Checkpoints are written below
`models/target_discriminators/squeezenet_attention/`. The normal standard
baseline runner discovers them and writes separate
`plots/discriminator/squeezenet_attention/` curves.

The main trainer saves only the inner torchvision model weights:

```text
results/weather_discriminator_<model_name>_<variable_tag>_lightning.pth
```

K-fold training writes:

```text
results/kfold_checkpoints/discriminator_<model_name>_<variable_tag>_exclude_<model>.pth
results/kfold_checkpoints/discriminator_<model_name>_<variable_tag>_full_pool.pth
```

Temporal holdout training writes:

```text
results/temporal_holdout_checkpoints/discriminator_<model_name>_<variable_tag>_temporal_<model>.pth
```

Because normalization statistics are recomputed from the configured real-data
training split, evaluation should use the same config that was used for training
unless the change is intentional.

## Notes And Caveats

- The default config expects data under `./data` unless `DATA_DIR` is set.
  Override `data_dir` or individual file paths for other layouts.
- `scripts/train_discriminator.py` validates configured train/test time splits before
  training.
- Analysis logits are not calibrated probabilities. They are most useful for
  relative comparisons under the same trained discriminator and normalization
  setup.
- The k-fold workflow assumes the configured neural forecast files can be
  resolved by filename convention. For leave-one-model-out folds, only the
  held-out model is removed from the fake training pool.
- K-fold training uses the union of the remaining fake-file date spans for fake
  samples and the complement of that union for ERA5/reference samples.
