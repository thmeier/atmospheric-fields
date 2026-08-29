# Atmospheric Field Discriminator

This directory contains an experimental discriminator pipeline for assigning
realism scores to atmospheric fields. The main discriminator learns a binary
classification task: ERA5/reference fields are labeled real, while model
forecasts and optional synthetic corruptions are labeled fake. The raw logit is
then used to derive a realism score in the evaluation and analysis scripts.

## Directory Structure

- `conf/config.yaml`: default Hydra configuration for data paths, variables,
  train/test ranges, model choice, augmentation, logging, and outputs.
- `conf/baseline_config.yaml`: tracked baseline configuration extending the
  base config with metrics, corruptions, and plotting options.
- `conf/baseline_pipeline.yaml`: pipeline orchestration settings (stages,
  histogram/moment matching, W&B tracking, temporal resampling).
- `conf/target_discriminator_baselines.yaml`: target-specific discriminator
  config (SqueezeNet, SFNO probes, interpretability, severity settings).
- `scripts/train_discriminator.py`: main training entrypoint. Trains one ResNet18
  or SqueezeNet discriminator and writes a `.pth` file under `output_dir`.
- `scripts/run_baseline_pipeline.py`: pipeline orchestrator — selects and runs
  stages in dependency order.
- `scripts/train_target_discriminator_baselines.py`: target discriminator
  training, IG interpretability, SFNO probes, logit histograms.
- `scripts/plot_standard_metric_baselines.py`: standard metric computation
  (SCWD, MMD, Wasserstein, etc.), evaluation, and plotting.
- `scripts/temporal_resampling_pipeline.py`: temporal resampling wrapper.
- `scripts/evaluate_discriminator.py`: computes test accuracy/loss and saves
  map panels for high-logit, uncertain, fooled, and obvious-fake samples.
- `scripts/corruptions.py`: tensor corruptions used during training and evaluation.
- `scripts/analysis_utils.py`: shared inference helpers for normalization, lead-time
  datasets, device selection, and checkpoint loading.
- `scripts/plot_bundles.py`: figure bundle infrastructure (PNG+PDF+NPZ), paper
  and dashboard profiles.
- `scripts/manage_pipeline_storage.py`: inventory, migrate, and prune pipeline
  storage.

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

### Standard Metric Baselines

The tracked baseline pipeline compares unpaired forecast or corrupted-field
distributions with ERA5. Temporal resampling is enabled by default. Learned
metrics train five independent critics with seven-day test windows on days
5–11, 9–15, 13–19, 17–23, and 20–26 of every month. Four complete days on each
side of a test window are excluded from that fold's training data. Fixed metrics
reuse those folds and add 45 deterministic within-month seven-day schedules.
Forecast membership is determined by valid time, and every forecast must have
an exact ERA5 match; there is no nearest-time fallback.
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
outputs record both `n_samples` and `pairwise_n_samples`. Standard curves show
the mean and 5th–95th percentiles over 10 temporal resamples. Learned curves show
the mean and full min–max envelope over five independently trained critics. No
within-test bootstrap is applied. The optional legacy bootstrap-null workflow
instead resamples disjoint ERA5 partitions to estimate a
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
/cluster/courses/pmlr/teams/team07/baseline_pipeline_runs/<pipeline-id>/
├── resolved_config.yaml
├── manifest.json
└── <variable-tag>/
    ├── data/                       (aggregate curves and all resample draws)
    │   ├── fixed_metric_draws.csv
    │   ├── fixed_metric_draws.npz
    │   ├── discriminator_metric_draws.csv
    │   ├── discriminator_metric_draws.npz
    │   └── resample_status.csv
    ├── models/target_discriminators/learned_XX/
    ├── training/learned_XX/
    └── plots/
        └── paper/
```

Every figure is accompanied by an NPZ data bundle and a title-less variant. The
title-less file is a hard link to the canonical NPZ rather than a duplicate. PNG
and NPZ are the defaults; PDF is opt-in with `plotting.save_pdf=true` (the paper
batch wrapper enables it). Paper mode uses manuscript-scale typography and places
outputs under `plots/paper/`.
SCWD anchor-response histograms are disabled by default; enable them with
`plotting.scwd_response_histograms=true`. Anchor maps and mean-response maps are
still generated.

The parent `data/` directory retains `metric_draws.csv`, long-form CSV and NPZ
arrays for learned/fixed draws, `split_manifest.csv.gz`, and raw
`discriminator_terms.csv.gz`. One invocation evaluates all resamples in-process; it does not create child
pipeline runs. All five learned folds retain checkpoints and scalar train/test
records, but only `pipeline.storage.canonical_diagnostic_fold` (default
`learned_04`) runs first and renders the large attribution, histogram, and
representation galleries. Checkpoints are uploaded once per fold as a coherent artifact.
These are uploaded as W&B evaluation artifacts. Every reverse-KL draw is checked
against the raw saved held-out ERA5-reference and candidate terms before
aggregation. Normalization remains fitted on the training split, while both
variational expectations are evaluated out of sample. For
example, count a model's null draws above its mean 12-hour score with:

```bash
python scripts/analyze_temporal_resamples.py \
  <run>/<variable-tag>/data/discriminator_metric_draws.csv \
  --architecture squeezenet --target GraphCast --coordinate 12
```

Set `temporal_resampling.enabled=false` for the original single-split debugging
workflow. Evaluation-only temporal runs accept
`temporal_resampling.input_run_dir=<prior-pipeline-run>` and resolve the matching
checkpoint separately for every learned fold.
Evaluation CSV/NetCDF artifacts and plot bundles are uploaded to W&B when the
corresponding pipeline upload switches are enabled. Online W&B working data is
created in node-local scratch and removed when the pipeline exits; offline mode
keeps it under the run directory. `wandb_runs.csv` is the persistent local index
of every dashboard run and its status.

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

### Temporal Logit Map Case Study

Create an interpretable held-out forecast case study at 24 h. The script ranks
test forecasts by their scalar realism logit, selects low/median/high examples,
and plots forecast, matching valid-time ERA5, and the pre-pooling spatial logit
map.

```bash
DATA_DIR=/cluster/courses/pmlr/teams/team07/data \
srun -A pmlr -t 00:30 python scripts/plot_temporal_logit_map_case_study.py
```

Use `case_study_model`, `case_study_lead_hour`, and
`case_study_logit_quantiles` in `conf/config.yaml` (or Hydra overrides) to
choose the temporal discriminator, lead, and examples.

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

Inspect storage or safely migrate an old run with dry-run-first commands:

```bash
python scripts/manage_pipeline_storage.py inventory --home "$HOME"
python scripts/manage_pipeline_storage.py migrate OLD_RUN TEAM_RUNS
python scripts/manage_pipeline_storage.py migrate OLD_RUN TEAM_RUNS --apply --delete-source
LEGACY_WANDB=/home/yelberkennou/atmospheric-fields/Discriminator/wandb
python scripts/manage_pipeline_storage.py clean-wandb --home "$HOME" --legacy-root "$LEGACY_WANDB"
python scripts/manage_pipeline_storage.py clean-wandb --home "$HOME" --legacy-root "$LEGACY_WANDB" --apply
```

Migration copies to a partial destination, verifies every file by size and SHA-256,
and only then activates it. Cleanup refuses to run while W&B processes are active.

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

It uses two learned folds, three fixed resamples, one epoch, 32 train/evaluation
samples, GraphCast, Gaussian blur and field splice, and the two inexpensive moment metrics, while still traversing every pipeline
stage and exercising online W&B uploads and aggregate plotting.

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

Moment matching is also disabled by default and is mutually exclusive with
histogram matching. Set `moment_matching.enabled=true` and include
`fit_moment_matching` to fit an ordinary pooled-grid-cell mean and standard
deviation for every variable, target, and lead/severity using training data only.
The frozen affine correction is applied to fake fields in physical units before
standardization. Evaluation-only runs provide
`pipeline.input_moment_matching_dir=/absolute/path/to/data/moment_matching`.
Each temporal resample retains its own fit artifact beneath the parent run.

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

By default, `pipeline.runs_dir=null` resolves first from
`PIPELINE_RUNS_DIR`. The Slurm wrappers default that variable to
`/work/scratch/$USER/baseline_pipeline_runs`, since `/cluster/courses` is
read-only on compute nodes. Direct invocations without the variable retain the
legacy fallback beside `DATA_DIR`. Each atomic manifest also
records total bytes and bytes by file suffix. The pipeline warns at 5 GiB and
stops at 10 GiB by default. The pipeline manifest and resolved configuration
live directly under the pipeline-run directory. Evaluation-only and plotting-only invocations create
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

Normalization statistics are recomputed from the configured real-data training
split; evaluation should use the same config that was used for training.

## Notes And Caveats

- The default config expects data under `./data` unless `DATA_DIR` is set.
  Override `data_dir` or individual file paths for other layouts.
- `scripts/train_discriminator.py` validates configured train/test time splits before
  training.
- Analysis logits are not calibrated probabilities. They are most useful for
  relative comparisons under the same trained discriminator and normalization
  setup.
