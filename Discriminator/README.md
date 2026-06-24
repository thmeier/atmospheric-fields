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

Train temporal holdout discriminators, one per forecast model with matching
train/test-period files:

```bash
python scripts/train_temporal_holdout.py
```

### Temporal Holdout Experiment

The temporal holdout experiment is configured by `conf/config.yaml`. It trains
one discriminator per forecast model using that model's training-period forecast
file as fake data and ERA5 from `train_real_range` as real data. It then
evaluates each discriminator on the same forecast model's test-period file,
using `test_fake_range` and the configured real test ranges. With the default
multi-field config, checkpoint names use the `all_fields` tag.

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
forecast model files are used. ERA5/reference data is not folded by model and,
for these k-fold runs, should be understood as using the whole available ERA5
data range configured for the experiment.

The full-pool k-fold discriminator keeps all neural forecast model files in the
fake training pool and is used for numerical-model comparisons.

Because the k-fold setup uses the whole available ERA5/reference range, its
configured train and test time ranges can overlap. Standard test accuracy/loss
from `scripts/evaluate_discriminator.py` should therefore not be interpreted as
independent held-out metrics for these k-fold checkpoints. The meaningful
holdout axis is the omitted forecast model, not a disjoint ERA5 time split.

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
- `train_real_range`, `train_fake_range`: training time splits.
- `test_real_ranges`, `test_fake_range`: holdout/test time splits.
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
  held-out model is removed from the fake training pool; the remaining model
  files and the configured full ERA5/reference range are used.
- K-fold train/test time ranges may overlap, so test accuracy/loss is not a
  meaningful held-out metric for those checkpoints.
