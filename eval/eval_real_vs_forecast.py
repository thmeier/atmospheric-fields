"""Evaluate latent-space FID and MMD between ERA5 reanalysis and 24h forecast products.

For each of two self-supervised encoders (MAE-twin, I-JEPA-twin) this script:
  1. Pairs ERA5 samples to forecast valid times (nearest-neighbour in time).
  2. Extracts latent features from each encoder.
  3. Computes Fréchet Distance and MMD between ERA5 and each forecast (Pangu, GraphCast).
  4. Saves a metric bar chart and a joint-PCA scatter plot.

Usage (local, smoke test):
    /opt/miniconda3/envs/pmlr/bin/python eval/eval_real_vs_forecast.py --local --n-samples 50

Usage (local, full):
    /opt/miniconda3/envs/pmlr/bin/python eval/eval_real_vs_forecast.py --local --n-samples 500
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import scipy.linalg
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils.dataset import AtmosphereDataset
from utils.model_io import build_model, checkpoint_path, load_model_checkpoint
from utils.features import extract_features_for_loader
from utils.temporal import IN_CHANS_BY_MODE, compose_temporal_input, derive_delta_steps

# ---------------------------------------------------------------------------
# Default paths (overridable via CLI flags below)
# ---------------------------------------------------------------------------

LOCAL_ERA5_PATH      = Path("data/test_data_local.nc")
LOCAL_PANGU_PATH     = Path("data/pangu_surface_2020_lead24h.nc")
LOCAL_GRAPHCAST_PATH = Path("data/graphcast_surface_2020_lead24h.nc")

CLUSTER_ERA5_PATH      = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
CLUSTER_PANGU_PATH     = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/pangu_surface_2020_lead24h.nc")
CLUSTER_GRAPHCAST_PATH = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/graphcast_surface_2020_lead24h.nc")

MODEL_COLORS  = {"mae": "#2196F3", "ijepa": "#FF5722"}
MODEL_LABELS  = {"mae": "MAE (twin)", "ijepa": "I-JEPA (twin)"}
SOURCE_COLORS = {"era5": "#4CAF50", "pangu": "#9C27B0", "graphcast": "#FF9800"}
SOURCE_LABELS = {"era5": "ERA5", "pangu": "Pangu-24h", "graphcast": "GraphCast-24h"}
SOURCE_HATCHES = {"pangu": "", "graphcast": "//"}

# ---------------------------------------------------------------------------
# Metric functions (mirrors eval_distances.py to keep this script self-contained)
# ---------------------------------------------------------------------------

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(f"Warning: Imaginary FD component {np.max(np.abs(covmean.imag)):.4f}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def mmd_rbf(X, Y, gamma=None):
    XX = torch.cdist(X, X, p=2) ** 2
    YY = torch.cdist(Y, Y, p=2) ** 2
    XY = torch.cdist(X, Y, p=2) ** 2
    if gamma is None:
        diag_mask = ~torch.eye(XX.shape[0], dtype=torch.bool)
        dists = torch.cat([XX[diag_mask], YY[diag_mask], XY.flatten()])
        gamma = 1.0 / (2.0 * torch.median(dists).item() + 1e-6)
    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)
    N, M = X.shape[0], Y.shape[0]
    K_XX.fill_diagonal_(0)
    K_YY.fill_diagonal_(0)
    return float(K_XX.sum() / (N * (N - 1)) + K_YY.sum() / (M * (M - 1)) - 2 * K_XY.mean())

# ---------------------------------------------------------------------------
# Time-aligned index building
# ---------------------------------------------------------------------------

def _read_times(nc_path):
    """Read the 'time' dimension of a NetCDF file as a pd.DatetimeIndex."""
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path, decode_times=True)
        raw = ds.time.values
        ds.close()
        if hasattr(raw[0], "strftime"):  # cftime objects
            return pd.DatetimeIndex([pd.Timestamp(str(t)) for t in raw])
        return pd.DatetimeIndex(raw)
    except Exception:
        # Fallback: netCDF4 directly
        from netCDF4 import Dataset as NC4, num2date
        with NC4(nc_path) as ds:
            tvar = ds.variables["time"]
            cftimes = num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"))
        return pd.DatetimeIndex([
            pd.Timestamp(t.year, t.month, t.day, t.hour, t.minute, t.second)
            for t in cftimes
        ])


def build_paired_indices(era5_path, forecast_path, max_dt_hours=6, prior_hours=None):
    """Return paired indices aligned by valid time.

    For each forecast valid time `ft`:
      - locate the nearest ERA5 timestamp to `ft` (present, `era5_idx`).
      - if `prior_hours` is given, also locate the nearest ERA5 timestamp to
        `ft − prior_hours` (prior, `era5_prior_idx`). Drop the pair if either
        match is > `max_dt_hours` away.

    Returns:
        (era5_idx_list, fc_idx_list) when prior_hours is None.
        (era5_idx_list, fc_idx_list, era5_prior_idx_list) when prior_hours is set.
    """
    era5_times = _read_times(era5_path)
    fc_times   = _read_times(forecast_path)

    era5_idx_list, fc_idx_list, era5_prior_idx_list = [], [], []
    skipped = 0
    skipped_prior = 0
    max_dt = pd.Timedelta(hours=max_dt_hours)
    prior_offset = pd.Timedelta(hours=prior_hours) if prior_hours is not None else None

    for fi, ft in enumerate(fc_times):
        delta = np.abs(era5_times - ft)
        ei = int(delta.argmin())
        if delta[ei] > max_dt:
            skipped += 1
            continue

        if prior_offset is not None:
            target_prior = ft - prior_offset
            delta_p = np.abs(era5_times - target_prior)
            ei_p = int(delta_p.argmin())
            if delta_p[ei_p] > max_dt:
                skipped_prior += 1
                continue
            era5_prior_idx_list.append(ei_p)

        era5_idx_list.append(ei)
        fc_idx_list.append(fi)

    if prior_offset is not None:
        print(f"  {len(era5_idx_list)} paired samples "
              f"({len(fc_times)} forecast times, {skipped} skipped on present, "
              f"{skipped_prior} skipped on prior (Δt={prior_hours}h), tolerance ±{max_dt_hours}h)")
        return era5_idx_list, fc_idx_list, era5_prior_idx_list

    print(f"  {len(era5_idx_list)} paired samples "
          f"({len(fc_times)} forecast times, {skipped} outside {max_dt_hours}h tolerance, "
          f"range {fc_times[0].date()} – {fc_times[-1].date()})")
    return era5_idx_list, fc_idx_list


def cap_pairs(era5_idx, fc_idx, n, rng):
    """Shuffle and cap paired index lists to min(n, len)."""
    total = len(era5_idx)
    if n >= total:
        return era5_idx, fc_idx
    perm = rng.permutation(total)[:n].tolist()
    return [era5_idx[i] for i in perm], [fc_idx[i] for i in perm]


def split_pairs(era5_idx, fc_idx, n, rng, era5_prior_idx=None):
    """Shuffle paired indices and split into two disjoint halves of size n each.

    Half A's (ERA5, forecast) pairs drive the model comparison; half B's ERA5
    indices act as the within-distribution baseline partner. Both halves come
    from the same time-paired pool (00/12 UTC for Pangu/GraphCast), so the
    baseline and forecast comparisons share the same diurnal/seasonal coverage.

    If `era5_prior_idx` is provided (temporal-pair eval), the same shuffle/split
    is applied to it, yielding both the present and prior ERA5 indices for each
    half.

    Returns:
        Non-temporal: (era5_a, fc_a, era5_b)
        Temporal:     (era5_a, fc_a, era5_b, era5_prior_a, era5_prior_b)
                      where each *_prior_* gives the ERA5 t-Δt index for that
                      half.
    """
    total = len(era5_idx)
    n_actual = min(n, total // 2)
    perm = rng.permutation(total).tolist()
    a_idx = perm[:n_actual]
    b_idx = perm[n_actual:2 * n_actual]
    if era5_prior_idx is None:
        return (
            [era5_idx[i] for i in a_idx],
            [fc_idx[i]   for i in a_idx],
            [era5_idx[i] for i in b_idx],
        )
    return (
        [era5_idx[i] for i in a_idx],
        [fc_idx[i]   for i in a_idx],
        [era5_idx[i] for i in b_idx],
        [era5_prior_idx[i] for i in a_idx],
        [era5_prior_idx[i] for i in b_idx],
    )


class TemporalPairDataset(torch.utils.data.Dataset):
    """Eval-time wrapper that builds (X_{t-Δt}, X_t) composite inputs from two
    source datasets at matched absolute indices.

    The prior side always comes from ERA5 (`prior_ds`). The present side comes
    from `present_ds`, which may be ERA5 (real) or a forecast file (fake).
    Both source datasets are constructed with `split="all"` so absolute indices
    are valid.
    """

    def __init__(self, prior_ds, present_ds, prior_idx, present_idx,
                 mode, abs_stats, diff_stats=None):
        if len(prior_idx) != len(present_idx):
            raise ValueError(
                f"prior_idx (len {len(prior_idx)}) and present_idx (len {len(present_idx)}) must match"
            )
        if mode == "none":
            raise ValueError("TemporalPairDataset is only for temporal modes (diff/concat/phase).")
        self.prior_ds = prior_ds
        self.present_ds = present_ds
        self.prior_idx = list(prior_idx)
        self.present_idx = list(present_idx)
        self.mode = mode
        self.abs_mean = abs_stats[0]
        self.abs_std  = abs_stats[1]
        self.diff_mean = diff_stats[0] if diff_stats is not None else None
        self.diff_std  = diff_stats[1] if diff_stats is not None else None

    def __len__(self):
        return len(self.prior_idx)

    def __getitem__(self, i):
        prior_raw   = self.prior_ds.read_raw(self.prior_idx[i])
        present_raw = self.present_ds.read_raw(self.present_idx[i])
        sample = compose_temporal_input(
            present_raw, prior_raw, self.mode,
            self.abs_mean, self.abs_std, self.diff_mean, self.diff_std,
        )
        return torch.from_numpy(sample)

# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def compute_distances(z_real, z_fake):
    """z_real, z_fake: (N, D) torch tensors (on CPU).
    Returns dict with 'fid' and 'mmd' keys."""
    z_real_np = z_real.numpy()
    z_fake_np = z_fake.numpy()
    mu_r, sigma_r = np.mean(z_real_np, axis=0), np.cov(z_real_np, rowvar=False)
    mu_f, sigma_f = np.mean(z_fake_np, axis=0), np.cov(z_fake_np, rowvar=False)
    fid = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
    mmd = mmd_rbf(z_real, z_fake)
    return {"fid": fid, "mmd": mmd}


def compute_era5_self_baseline(z_era5_a, z_era5_b):
    """FID/MMD between two independent ERA5 samples (same N as forecast comparison).

    Gives the noise floor: what the metric reads when both sides come from the
    ERA5 distribution. Must use matched sample size to be comparable — FID is
    a biased estimator that grows with smaller N (rank-deficient covariance).
    """
    return compute_distances(z_era5_a, z_era5_b)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_metrics_table(results, models, n_pangu, n_gc, device, label_suffix=""):
    hdr = "=" * 80
    print(f"\n{hdr}")
    print("REAL vs 24h FORECAST — LATENT DISTANCES")
    print(hdr)
    print(f"N pairs (Pangu): {n_pangu}    N pairs (GraphCast): {n_gc}    Device: {device}")
    print()
    print(f"  {'Model':<40} {'Forecast':<14} {'FID':>10} {'MMD':>12}")
    print("  " + "-" * 78)
    for model_name in models:
        label = f"{MODEL_LABELS[model_name]}{label_suffix}"
        baseline = results[model_name].get("era5_self")
        if baseline:
            print(f"  {label:<40} {'ERA5 vs ERA5':<14} {baseline['fid']:>10.3f} {baseline['mmd']:>12.6f}  ← baseline")
        for source in ["pangu", "graphcast"]:
            d = results[model_name][source]
            print(f"  {label:<40} {SOURCE_LABELS[source]:<14} {d['fid']:>10.3f} {d['mmd']:>12.6f}")
        print()
    print(hdr)


def plot_metric_bars(results, models, plots_dir, run_tag):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sources = ["pangu", "graphcast"]
    x = np.arange(len(models))
    width = 0.35

    for ax, metric_key, metric_label in [
        (axes[0], "fid", "Fréchet Distance"),
        (axes[1], "mmd", "MMD"),
    ]:
        for si, source in enumerate(sources):
            vals = [results[m][source][metric_key] for m in models]
            bars = ax.bar(
                x + (si - 0.5) * width, vals, width=width,
                label=SOURCE_LABELS[source],
                color=[MODEL_COLORS[m] for m in models],
                hatch=SOURCE_HATCHES[source],
                edgecolor="white", linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                )

        # ERA5 self-baseline: one dashed horizontal line per model
        for mi, model_name in enumerate(models):
            baseline = results[model_name].get("era5_self")
            if baseline:
                bval = baseline[metric_key]
                x_left  = x[mi] - width
                x_right = x[mi] + width
                ax.hlines(bval, x_left, x_right,
                          colors=MODEL_COLORS[model_name], linestyles="dashed",
                          linewidth=1.5, label=f"ERA5 self ({MODEL_LABELS[model_name]})")
                ax.text(x_right + 0.03, bval, f"{bval:.3f}",
                        va="center", ha="left", fontsize=7,
                        color=MODEL_COLORS[model_name])

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models])
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label}: ERA5 vs 24h Forecast")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Forecast / Baseline", loc="upper right", fontsize=8)

    if len(models) > 1:
        color_handles = [Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in models]
        fig.legend(handles=color_handles, title="Model", loc="lower center",
                   ncol=2, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(f"ERA5 vs 24h Forecast — Latent Distance\nRun: {run_tag}", fontsize=12)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"metric_bars_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_pca_scatter(features_by_model, results, models, plots_dir, run_tag, model_label_suffix=""):
    """features_by_model: {model: {"era5": ndarray, "pangu": ndarray, "graphcast": ndarray}}"""
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        feats = features_by_model[model_name]
        all_f = np.vstack([feats["era5"], feats["pangu"], feats["graphcast"]])
        n_e, n_p = len(feats["era5"]), len(feats["pangu"])

        # Joint PCA via SVD
        mean = all_f.mean(axis=0, keepdims=True)
        Xc = all_f - mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt[:2].T  # (N_total, 2)

        era5_p  = proj[:n_e]
        pangu_p = proj[n_e:n_e + n_p]
        gc_p    = proj[n_e + n_p:]

        for pts, src in [(era5_p, "era5"), (pangu_p, "pangu"), (gc_p, "graphcast")]:
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=SOURCE_COLORS[src], alpha=0.4, s=12, label=SOURCE_LABELS[src])

        fid_p = results[model_name]["pangu"]["fid"]
        fid_g = results[model_name]["graphcast"]["fid"]
        ax.text(0.02, 0.98,
                f"FID Pangu: {fid_p:.2f}\nFID GCast: {fid_g:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(f"{MODEL_LABELS[model_name]}{model_label_suffix}")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.grid(True, alpha=0.2)
        ax.legend(markerscale=2, fontsize=9)

    fig.suptitle(f"Latent PCA: ERA5 vs 24h Forecasts\nRun: {run_tag}", fontsize=12)
    fig.tight_layout()
    out = plots_dir / f"pca_scatter_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare ERA5 vs forecast latent distributions (MAE + I-JEPA).")
    parser.add_argument("--local", action="store_true",
                        help="Use local data paths (data/*.nc). Default = cluster paths.")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Max paired samples per (model, source) pair (default: 500).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-size", choices=["default", "twin", "tiny", "small"], default="twin",
                        help="Model size config to build for both MAE and IJEPA (default: twin).")
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="Override encoder embed_dim for both models (matches train flag).")
    parser.add_argument("--variant", type=str, default=None,
                        help="Checkpoint variant suffix (e.g. 'shared-targets').")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory holding checkpoints + stats; plots saved under <dir>/plots/real_vs_forecast. "
                             "If unset, falls back to checkpoints/ + plots/real_vs_forecast/.")
    parser.add_argument("--era5-path", type=str, default=None,
                        help="Override ERA5 NetCDF path.")
    parser.add_argument("--pangu-path", type=str, default=None,
                        help="Override Pangu NetCDF path.")
    parser.add_argument("--graphcast-path", type=str, default=None,
                        help="Override GraphCast NetCDF path.")
    parser.add_argument("--pooling", choices=["mean", "max", "concat"], default=None,
                        help="Pooling mode for extract_features. Sets EXTRACT_FEATURES_POOLING. "
                             "Defaults to the env var if set, otherwise 'mean'.")
    parser.add_argument("--mae-only", action="store_true",
                        help="Only run MAE (skip I-JEPA checkpoint requirement).")
    parser.add_argument("--temporal-mode", choices=["none", "diff", "concat", "phase"], default="none",
                        help="Temporal-pair eval mode. Must match the trained checkpoint's mode.")
    parser.add_argument("--delta-hours", type=int, default=24,
                        help="Δt in hours for the temporal pair (default 24).")
    args = parser.parse_args()

    if args.pooling is not None:
        os.environ["EXTRACT_FEATURES_POOLING"] = args.pooling

    era5_path = Path(args.era5_path) if args.era5_path else (
        LOCAL_ERA5_PATH if args.local else CLUSTER_ERA5_PATH)
    pangu_path = Path(args.pangu_path) if args.pangu_path else (
        LOCAL_PANGU_PATH if args.local else CLUSTER_PANGU_PATH)
    graphcast_path = Path(args.graphcast_path) if args.graphcast_path else (
        LOCAL_GRAPHCAST_PATH if args.local else CLUSTER_GRAPHCAST_PATH)

    stats_dir = Path(args.output_dir) if args.output_dir else Path("checkpoints")
    models_to_run = ["mae"] if args.mae_only else ["mae", "ijepa"]

    # Compose the checkpoint variant: a CLI-supplied --variant takes precedence,
    # otherwise temporal modes fall back to tm-<mode>. This must match what the
    # training scripts wrote.
    effective_variant = args.variant
    if effective_variant is None and args.temporal_mode != "none":
        effective_variant = f"tm-{args.temporal_mode}"

    ckpt_paths = {
        m: checkpoint_path(m, args.model_size, stats_dir, variant=effective_variant, embed_dim=args.embed_dim)
        for m in models_to_run
    }

    for p in [era5_path, pangu_path, graphcast_path] + [ckpt_paths[m] for m in models_to_run]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    temporal = args.temporal_mode != "none"
    prior_hours = args.delta_hours if temporal else None

    print("Building time-aligned index pairs...")
    print("  Pangu:")
    pangu_paired = build_paired_indices(era5_path, pangu_path, prior_hours=prior_hours)
    print("  GraphCast:")
    gc_paired    = build_paired_indices(era5_path, graphcast_path, prior_hours=prior_hours)

    if temporal:
        era5_p_idx_all, pangu_idx_all, era5_p_prior_all = pangu_paired
        era5_g_idx_all, gc_idx_all,    era5_g_prior_all = gc_paired
        era5_p_idx, pangu_idx, era5_base_idx, era5_p_prior, era5_base_prior = split_pairs(
            era5_p_idx_all, pangu_idx_all, args.n_samples, rng, era5_prior_idx=era5_p_prior_all)
        era5_g_idx, gc_idx, _, era5_g_prior, _ = split_pairs(
            era5_g_idx_all, gc_idx_all, args.n_samples, rng, era5_prior_idx=era5_g_prior_all)
    else:
        era5_p_idx_all, pangu_idx_all = pangu_paired
        era5_g_idx_all, gc_idx_all    = gc_paired
        era5_p_idx, pangu_idx, era5_base_idx = split_pairs(
            era5_p_idx_all, pangu_idx_all, args.n_samples, rng)
        era5_g_idx, gc_idx, _ = split_pairs(
            era5_g_idx_all, gc_idx_all, args.n_samples, rng)
        era5_p_prior = era5_g_prior = era5_base_prior = None

    n_pangu = len(pangu_idx)
    n_gc    = len(gc_idx)
    n_base  = len(era5_base_idx)

    if n_base < args.n_samples:
        print(f"\nNote: requested n_samples={args.n_samples} but paired pool only "
              f"supports {n_base} per half. Forecast and baseline both run at N={n_base}.")
    print(f"\nUsing {n_pangu} Pangu pairs, {n_gc} GraphCast pairs, "
          f"and {n_base} ERA5 baseline samples (disjoint Pangu-paired half, "
          f"matched diurnal coverage).")

    stats = (
        np.load(stats_dir / "data_mean.npy"),
        np.load(stats_dir / "data_std.npy"),
    )
    diff_stats = None
    if args.temporal_mode in ("diff", "phase"):
        dm_path = stats_dir / f"diff_mean_dt{args.delta_hours}h.npy"
        ds_path = stats_dir / f"diff_std_dt{args.delta_hours}h.npy"
        if not (dm_path.exists() and ds_path.exists()):
            raise FileNotFoundError(
                f"Temporal mode '{args.temporal_mode}' requires {dm_path.name} + {ds_path.name} "
                f"in {stats_dir}. Re-run training with this temporal mode to produce them."
            )
        diff_stats = (np.load(dm_path), np.load(ds_path))

    # On the cluster, ERA5 has ~27800 samples (~13GB) and we only need ~500 — eager
    # loading would OOM. Locally the dataset is small enough to load eagerly.
    era5_lazy = not args.local
    print(f"\nLoading datasets (ERA5 lazy={era5_lazy}, forecasts eager; using ERA5 normalization stats)...")
    # For temporal eval we use the source datasets only as raw-readers
    # (via .read_raw), so we set temporal_mode='none' here — the composite
    # happens inside TemporalPairDataset.
    era5_ds  = AtmosphereDataset(era5_path,      split="all", stats=stats, lazy=era5_lazy)
    pangu_ds = AtmosphereDataset(pangu_path,     split="all", stats=stats, lazy=False)
    gc_ds    = AtmosphereDataset(graphcast_path, split="all", stats=stats, lazy=False)

    loader_kw = dict(batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, pin_memory=device.type == "cuda")

    # Feature dict: {model: {source: tensor (N, D)}}
    all_feats = {}
    results   = {}

    label_suffix = ""
    if args.embed_dim is not None:
        label_suffix += f" d{args.embed_dim}"
    if args.variant:
        label_suffix += f" [{args.variant}]"

    in_chans = IN_CHANS_BY_MODE[args.temporal_mode]

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {MODEL_LABELS[model_name]}{label_suffix}")
        print(f"{'='*60}")

        model = build_model(model_name, device, args.model_size, embed_dim=args.embed_dim,
                            in_chans=in_chans)
        ckpt  = ckpt_paths[model_name]
        print(f"Loading checkpoint: {ckpt}")
        model = load_model_checkpoint(model_name, model, ckpt, device)
        model.eval()

        # Verify the checkpoint's training config matches the requested eval mode.
        # Some early/static checkpoints predate this field — skip the check then.
        try:
            ckpt_blob = torch.load(ckpt, map_location="cpu")
            ckpt_cfg = ckpt_blob.get("config", {}) if isinstance(ckpt_blob, dict) else {}
            ckpt_tm = ckpt_cfg.get("temporal_mode", "none")
            ckpt_dh = ckpt_cfg.get("delta_hours", None)
        except Exception:
            ckpt_tm, ckpt_dh = "none", None
        if ckpt_tm != args.temporal_mode:
            raise RuntimeError(
                f"Checkpoint temporal_mode={ckpt_tm!r} but eval requested {args.temporal_mode!r}. "
                f"Use the matching checkpoint or pass the right --temporal-mode."
            )
        if args.temporal_mode != "none" and ckpt_dh is not None and ckpt_dh != args.delta_hours:
            raise RuntimeError(
                f"Checkpoint delta_hours={ckpt_dh} but eval requested {args.delta_hours}. "
                f"Diff stats and pair offset must match training."
            )

        if args.temporal_mode == "none":
            loaders = {
                "era5_p":    DataLoader(Subset(era5_ds,  era5_p_idx),    **loader_kw),
                "pangu":     DataLoader(Subset(pangu_ds, pangu_idx),     **loader_kw),
                "era5_g":    DataLoader(Subset(era5_ds,  era5_g_idx),    **loader_kw),
                "graphcast": DataLoader(Subset(gc_ds,    gc_idx),        **loader_kw),
                "era5_base": DataLoader(Subset(era5_ds,  era5_base_idx), **loader_kw),
            }
        else:
            # Temporal eval: present-side may be ERA5 (real) or a forecast.
            # Prior side is always ERA5 at the index t − Δt.
            def _tpd(prior_ds, present_ds, prior_idx, present_idx):
                return TemporalPairDataset(
                    prior_ds, present_ds, prior_idx, present_idx,
                    args.temporal_mode, abs_stats=stats, diff_stats=diff_stats,
                )
            loaders = {
                "era5_p":    DataLoader(_tpd(era5_ds, era5_ds,  era5_p_prior,    era5_p_idx),    **loader_kw),
                "pangu":     DataLoader(_tpd(era5_ds, pangu_ds, era5_p_prior,    pangu_idx),     **loader_kw),
                "era5_g":    DataLoader(_tpd(era5_ds, era5_ds,  era5_g_prior,    era5_g_idx),    **loader_kw),
                "graphcast": DataLoader(_tpd(era5_ds, gc_ds,    era5_g_prior,    gc_idx),        **loader_kw),
                "era5_base": DataLoader(_tpd(era5_ds, era5_ds,  era5_base_prior, era5_base_idx), **loader_kw),
            }

        feats = {}
        for name, loader in loaders.items():
            print(f"  Extracting features: {name} ({len(loader.dataset)} samples)...")
            feats[name] = extract_features_for_loader(model, loader, device)

        # Both baseline halves come from the same Pangu-paired time pool at matched N,
        # so this is a true within-distribution noise floor.
        print(f"  Computing ERA5 self-baseline (N={len(era5_base_idx)} vs N={n_pangu})...")
        results[model_name] = {
            "era5_self": compute_era5_self_baseline(feats["era5_p"], feats["era5_base"]),
            "pangu":     compute_distances(feats["era5_p"], feats["pangu"]),
            "graphcast": compute_distances(feats["era5_g"], feats["graphcast"]),
        }
        all_feats[model_name] = {
            "era5":      feats["era5_p"].numpy(),   # representative ERA5 distribution
            "pangu":     feats["pangu"].numpy(),
            "graphcast": feats["graphcast"].numpy(),
        }

    print_metrics_table(results, models_to_run, n_pangu, n_gc, device, label_suffix=label_suffix)

    tag_parts = [args.model_size]
    if args.embed_dim is not None:
        tag_parts.append(f"d{args.embed_dim}")
    if args.temporal_mode != "none":
        tag_parts.append(f"tm-{args.temporal_mode}")
    if args.variant:
        tag_parts.append(args.variant)
    tag_parts.append(f"n{n_pangu}_seed{args.seed}")
    tag_parts.append(f"pool-{os.environ.get('EXTRACT_FEATURES_POOLING', 'mean').lower()}")
    run_tag = "_".join(tag_parts)

    if args.output_dir:
        plots_dir = Path(args.output_dir) / "plots" / "real_vs_forecast"
    else:
        plots_dir = Path("plots/real_vs_forecast")
    plot_metric_bars(results, models_to_run, plots_dir, run_tag)
    plot_pca_scatter(all_feats, results, models_to_run, plots_dir, run_tag, model_label_suffix=label_suffix)


if __name__ == "__main__":
    main()
