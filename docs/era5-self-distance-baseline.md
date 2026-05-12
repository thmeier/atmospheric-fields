# ERA5 Self-Distance Baseline and Forecast Comparison

## Motivation

Our project trains self-supervised encoders (MAE and I-JEPA) on ERA5 reanalysis data and uses the learned latent space to define a realism metric for atmospheric fields.
The metric computes Fréchet Distance (FID) and Maximum Mean Discrepancy (MMD) between two sets of latent vectors — a reference distribution and a test distribution.
A high distance is meant to signal that the test fields are less realistic.

Before using these distances to evaluate forecast models, we need to answer a foundational question:

> **What does the metric look like when both inputs are drawn from the same real distribution?**

Without this baseline, a reported distance of, say, 40 FID between ERA5 and GraphCast is uninterpretable — we do not know whether 40 is large or small for this encoder and this data.
There are two concrete failure modes the baseline guards against:

1. **The metric may have high variance due to finite-sample noise.** With 250 or 500 latent vectors, covariance estimation is noisy, and FID can fluctuate substantially across random draws of the same dataset. If that variance is large relative to the ERA5 vs forecast gap, the metric is not reliable.
2. **The metric may be systematically non-zero even for identical distributions.** FID and MMD both have a finite-sample bias; knowing its typical magnitude tells us what threshold is meaningful.

The second question motivates a direct comparison with actual forecast distances:

> **Are ERA5 vs GraphCast / Pangu distances distinguishably larger than the ERA5 self-distance?**

This is the key validity test for the metric as a realism detector.

---

## What Was Implemented

A new evaluation script `eval/eval_era5_self_distance.py` that performs both experiments.

### Phase 1 — Bootstrap Null Distribution

**Algorithm:**

1. Load the full ERA5 dataset (all time steps, normalized with the training statistics from `checkpoints/`).
2. Run one forward pass through the encoder to extract latent features for every sample → a single `(N, D)` feature matrix.
3. Repeat `N_TRIALS` times:
   - Draw a random permutation of the `N` indices.
   - Take the first `n_per_split` indices as set A and the next `n_per_split` as set B (non-overlapping by construction).
   - Compute FID and MMD between A and B.
4. Report mean ± std over trials.

The single-pass extraction is important for efficiency: re-running the encoder for each of 20 trials would take 20× as long. Since all we need is to subsample the pre-computed matrix, each trial adds only the cost of a covariance computation.

**Output:**
- Console table: model × FID mean ± std × MMD mean ± std
- Plot: violin + jitter strip for each model, FID and MMD side by side (`plots/era5_self_distance/bootstrap_*.png`)

### Phase 2 — Forecast Comparison (optional)

When `--pangu-path` and/or `--graphcast-path` are provided:

1. Load the forecast NetCDF file.
2. Time-align the forecast valid times to ERA5 timestamps (nearest-neighbour, ≤ 6 h tolerance), producing paired index lists.
3. Cap to `2 × n_per_split` pairs (matching the sample count of each bootstrap split) and shuffle.
4. Extract ERA5 features for the paired ERA5 indices and forecast features for the forecast indices.
5. Compute a single FID + MMD for each (ERA5, forecast) pair.
6. Plot all comparisons together: ERA5 self-distance as a bar with ±1σ error bar, forecasts as solid bars.

**Output:**
- Console table: extended with ERA5 vs Pangu, ERA5 vs GraphCast rows
- Plot: grouped bar chart per model (`plots/era5_self_distance/comparison_*.png`)

---

## Implementation Notes

- **Self-contained:** The script copies metric functions (`calculate_frechet_distance`, `mmd_rbf`) and time-alignment helpers (`build_paired_indices`) locally, following the convention established by the other eval scripts.
- **ERA5 split:** `split="all"` is used to maximise the available pool for bootstrapping. The existing eval scripts use `split="val"` for the corruption study, but for a baseline of the null distribution the larger pool is preferable.
- **Paired sampling for forecasts:** Using `2 × n_per_split` paired samples keeps the forecast comparison on the same scale as the ERA5 self-split pairs, making FID/MMD values directly comparable in the bar chart.
- **Forecast data:** Pangu and GraphCast fields are from WeatherBench2 at 1.5° resolution, downloaded at 24 h lead time and normalised with the same ERA5 training statistics.

---

## Results

All results use the twin encoder configuration (embed_dim=192, depth=8), 20 bootstrap trials, 250 samples per split, seed=0.
The forecast datasets cover 2020 only (732 paired samples; 500 are used to match the split size).

### Phase 1 — ERA5 Self-Distance (Bootstrap)

#### Mean pooling

| Model | FID mean ± std | MMD mean ± std |
|---|---|---|
| MAE (twin) | 0.194 ± 0.021 | −0.000466 ± 0.001513 |
| I-JEPA (twin) | 0.081 ± 0.028 | −0.000903 ± 0.001396 |

#### Max pooling

| Model | FID mean ± std | MMD mean ± std |
|---|---|---|
| MAE (twin) | 7.634 ± 0.139 | −0.000099 ± 0.000556 |
| I-JEPA (twin) | 1.639 ± 0.069 | −0.000609 ± 0.000990 |

*Bootstrap distribution plots:*
![Bootstrap mean pooling](../plots/era5_self_distance/bootstrap_twin_t20_n250_seed0_pool-mean.png)
![Bootstrap max pooling](../plots/era5_self_distance/bootstrap_twin_t20_n250_seed0_pool-max.png)

---

### Phase 2 — ERA5 vs Forecast Distances

#### Mean pooling

| Model | Comparison | FID | MMD |
|---|---|---|---|
| MAE (twin) | ERA5 ↔ ERA5 (mean ± std) | 0.194 ± 0.021 | −0.000466 ± 0.001513 |
| MAE (twin) | ERA5 ↔ Pangu-24h | 0.018 | −0.000639 |
| MAE (twin) | ERA5 ↔ GraphCast-24h | 0.018 | −0.000412 |
| I-JEPA (twin) | ERA5 ↔ ERA5 (mean ± std) | 0.081 ± 0.028 | −0.000903 ± 0.001396 |
| I-JEPA (twin) | ERA5 ↔ Pangu-24h | 0.001 | −0.001636 |
| I-JEPA (twin) | ERA5 ↔ GraphCast-24h | 0.002 | −0.001618 |

#### Max pooling

| Model | Comparison | FID | MMD |
|---|---|---|---|
| MAE (twin) | ERA5 ↔ ERA5 (mean ± std) | 7.634 ± 0.139 | −0.000099 ± 0.000556 |
| MAE (twin) | ERA5 ↔ Pangu-24h | 0.528 | −0.001029 |
| MAE (twin) | ERA5 ↔ GraphCast-24h | 0.433 | −0.000870 |
| I-JEPA (twin) | ERA5 ↔ ERA5 (mean ± std) | 1.684 ± 0.105 | 0.000089 ± 0.001466 |
| I-JEPA (twin) | ERA5 ↔ Pangu-24h | 0.061 | −0.001585 |
| I-JEPA (twin) | ERA5 ↔ GraphCast-24h | 0.053 | −0.001546 |

*Comparison bar charts:*
![Comparison mean pooling](../plots/era5_self_distance/comparison_twin_t20_n250_seed0_pool-mean.png)
![Comparison max pooling](../plots/era5_self_distance/comparison_twin_t20_n250_seed0_pool-max.png)

---

## Interpretation

### The headline finding: forecasts appear closer to ERA5 than ERA5 is to itself

The central result is that **ERA5 ↔ forecast FID is consistently smaller than the ERA5 self-distance baseline** — by roughly 10× under mean pooling (MAE: 0.018 vs 0.194) and consistently so under max pooling too (MAE: 0.53 vs 7.63). The metric in its current form **cannot distinguish 24 h forecasts from real ERA5** — it places them strictly within the null distribution.

### Why the forecast FID is lower than the self-distance

There are two plausible explanations, and both are likely contributing:

**1. Temporal alignment confound (probably dominant).**
The ERA5 bootstrap draws randomly from 20 years of data (2004–2023), so each 250-sample split spans the full range of inter-annual climate variability — ENSO years, blocking events, anomalous winters. Two such splits will differ in their representation of rare climate states, which inflates the measured FID.
The forecast comparison is fundamentally different: both sides are locked to 2020, and ERA5 samples are chosen to match the exact forecast valid times. This means we are comparing ERA5-2020 to forecasts-2020, not ERA5-20yr to forecasts-2020. The 2020-constrained ERA5 distribution is much more compact, and the FID will be correspondingly smaller — possibly regardless of how realistic the forecasts are.

**2. The encoders are insensitive to forecast artifacts at 24 h lead.**
Pangu and GraphCast at 24 h lead time are state-of-the-art: their RMSE is on the order of 1°C for temperature and ~1 m/s for wind. The differences that remain — slight spectral power deficits at small scales, marginal over-smoothing of fronts — are subtle, and a mean-pooled 192-dim latent trained purely on ERA5 reconstruction may not have learned to represent them.

### MMD gives no signal at all

MMD oscillates around zero for every comparison, including the ERA5 self-distance. The values are consistent with numerical noise rather than any systematic distributional difference. At n=250 and D=192, the RBF kernel with median-heuristic bandwidth is not a reliable discriminator.

### Max pooling makes things worse, not better

Max pooling inflates the null FID dramatically (7.6 vs 0.19 for MAE), but the forecast distances rise only modestly in comparison (0.53 vs 0.018). The ratio of null-to-forecast FID actually widens, making the pooling strategy even less useful. Max pooling captures extreme patch activations, which are highly sensitive to sampling noise at n=250 — the bootstrapped covariance is dominated by which rare extreme events happened to fall in each split.

### What this tells us about the metric

The current FID/MMD metric, as applied here, is not viable as a standalone realism score for comparing ERA5 to 24 h forecasts. The two most actionable paths forward are:

- **Disentangle the temporal confound** by bootstrapping within the same year (ERA5-2020 vs ERA5-2020 splits) to get a fair null baseline for the forecast comparison.
- **Train or fine-tune the encoder to be sensitive to forecast artifacts** — e.g., by including synthetic degradations that mimic known forecast biases (spectral truncation, smoothing), or by contrastive training on paired real/forecast fields.

---

## How to Run

### Local smoke test

Verifies the script runs end-to-end. The local dataset (`data/test_data_local.nc`) has ~128 samples, so `--n-per-split` must be ≤ 64.

```bash
# Phase 1 only, single model
python eval/eval_era5_self_distance.py \
    --local \
    --model mae \
    --n-trials 3 \
    --n-per-split 20

# Phase 1 only, both models
python eval/eval_era5_self_distance.py \
    --local \
    --model both \
    --n-trials 3 \
    --n-per-split 20

# Phase 1 + Phase 2 (requires local forecast files)
python eval/eval_era5_self_distance.py \
    --local \
    --model both \
    --n-trials 3 \
    --n-per-split 20 \
    --pangu-path data/pangu_surface_2020_lead24h.nc \
    --graphcast-path data/graphcast_surface_2020_lead24h.nc
```

Plots are saved to `plots/era5_self_distance/`.

### On the cluster

Phase 1 only (no forecast data needed):

```bash
python eval/eval_era5_self_distance.py \
    --model both \
    --n-trials 20 \
    --n-per-split 250
```

Phase 1 + Phase 2 with both forecast sources:

```bash
python eval/eval_era5_self_distance.py \
    --model both \
    --n-trials 20 \
    --n-per-split 250 \
    --pangu-path /cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/pangu_surface_2020_lead24h.nc \
    --graphcast-path /cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/graphcast_surface_2020_lead24h.nc
```

To use a non-default checkpoint directory (e.g. an experiment run saved elsewhere):

```bash
python eval/eval_era5_self_distance.py \
    --model both \
    --n-trials 20 \
    --n-per-split 250 \
    --output-dir /work/scratch/$USER/experiment-xyz
```

### Key arguments reference

| Argument | Default | Description |
|---|---|---|
| `--local` | off | Use small local dataset (~128 samples) |
| `--large-local` | off | Use 5-year local dataset |
| `--n-trials` | 20 | Number of bootstrap repetitions |
| `--n-per-split` | 250 | Samples per split half (need 2× ≤ total ERA5 samples) |
| `--model` | `both` | `mae`, `ijepa`, or `both` |
| `--model-size` | `twin` | Checkpoint config (must match the trained model) |
| `--embed-dim` | — | Override encoder embed_dim if a non-default size was trained |
| `--variant` | — | Checkpoint variant suffix (e.g. `shared-targets`) |
| `--seed` | 0 | Random seed for reproducibility |
| `--batch-size` | 32 | Batch size for feature extraction |
| `--pangu-path` | — | Path to Pangu NetCDF; enables Phase 2 for Pangu |
| `--graphcast-path` | — | Path to GraphCast NetCDF; enables Phase 2 for GraphCast |
| `--output-dir` | `checkpoints/` | Directory containing checkpoint and stats files; also controls plot output root |
