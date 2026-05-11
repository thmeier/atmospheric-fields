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

### Phase 1 — ERA5 Self-Distance (Bootstrap)

> **[PLACEHOLDER — run on cluster with `--n-trials 20 --n-per-split 250`]**

| Model | FID mean ± std | MMD mean ± std |
|---|---|---|
| MAE (twin) | `___ ± ___` | `___ ± ___` |
| I-JEPA (twin) | `___ ± ___` | `___ ± ___` |

*Bootstrap distribution plot:*
![Bootstrap distribution](../plots/era5_self_distance/bootstrap_twin_t20_n250_seed0_pool-mean.png)

---

### Phase 2 — ERA5 vs Forecast Distances

> **[PLACEHOLDER — run on cluster with Pangu and GraphCast paths]**

| Model | Comparison | FID | MMD |
|---|---|---|---|
| MAE (twin) | ERA5 ↔ ERA5 (mean) | `___` | `___` |
| MAE (twin) | ERA5 ↔ Pangu-24h | `___` | `___` |
| MAE (twin) | ERA5 ↔ GraphCast-24h | `___` | `___` |
| I-JEPA (twin) | ERA5 ↔ ERA5 (mean) | `___` | `___` |
| I-JEPA (twin) | ERA5 ↔ Pangu-24h | `___` | `___` |
| I-JEPA (twin) | ERA5 ↔ GraphCast-24h | `___` | `___` |

*Comparison bar chart:*
![Forecast comparison](../plots/era5_self_distance/comparison_twin_t20_n250_seed0_pool-mean.png)

**Key question to answer from the results:**
Are the ERA5 ↔ forecast bars clearly outside the ERA5 ↔ ERA5 mean ± std range, or do they fall within it?
If within, the metric cannot distinguish 24 h forecasts from real ERA5 at this sample size and encoder capacity.

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
