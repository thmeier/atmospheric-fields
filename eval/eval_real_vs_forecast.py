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


def build_paired_indices(era5_path, forecast_path, max_dt_hours=6):
    """Return (era5_idx_list, forecast_idx_list) aligned by valid time.

    For each forecast valid time the nearest ERA5 timestamp is located.
    Pairs where |Δt| > max_dt_hours are dropped.
    """
    era5_times = _read_times(era5_path)
    fc_times   = _read_times(forecast_path)

    era5_idx_list, fc_idx_list = [], []
    skipped = 0
    max_dt = pd.Timedelta(hours=max_dt_hours)

    for fi, ft in enumerate(fc_times):
        delta = np.abs(era5_times - ft)
        ei = int(delta.argmin())
        if delta[ei] <= max_dt:
            era5_idx_list.append(ei)
            fc_idx_list.append(fi)
        else:
            skipped += 1

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

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_metrics_table(results, n_pangu, n_gc, device, label_suffix=""):
    hdr = "=" * 80
    print(f"\n{hdr}")
    print("REAL vs 24h FORECAST — LATENT DISTANCES")
    print(hdr)
    print(f"N pairs (Pangu): {n_pangu}    N pairs (GraphCast): {n_gc}    Device: {device}")
    print()
    print(f"  {'Model':<40} {'Forecast':<14} {'FID':>10} {'MMD':>12}")
    print("  " + "-" * 78)
    for model_name in ["mae", "ijepa"]:
        label = f"{model_name.upper()} (twin){label_suffix}"
        for source in ["pangu", "graphcast"]:
            d = results[model_name][source]
            print(f"  {label:<40} {SOURCE_LABELS[source]:<14} {d['fid']:>10.3f} {d['mmd']:>12.6f}")
        print()
    print(hdr)


def plot_metric_bars(results, plots_dir, run_tag):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models  = ["mae", "ijepa"]
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
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models])
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label}: ERA5 vs 24h Forecast")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Forecast", loc="upper right")

    # Bottom legend for model → color mapping
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


def plot_pca_scatter(features_by_model, results, plots_dir, run_tag, model_label_suffix=""):
    """features_by_model: {model: {"era5": ndarray, "pangu": ndarray, "graphcast": ndarray}}"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model_name in zip(axes, ["mae", "ijepa"]):
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
    args = parser.parse_args()

    era5_path = Path(args.era5_path) if args.era5_path else (
        LOCAL_ERA5_PATH if args.local else CLUSTER_ERA5_PATH)
    pangu_path = Path(args.pangu_path) if args.pangu_path else (
        LOCAL_PANGU_PATH if args.local else CLUSTER_PANGU_PATH)
    graphcast_path = Path(args.graphcast_path) if args.graphcast_path else (
        LOCAL_GRAPHCAST_PATH if args.local else CLUSTER_GRAPHCAST_PATH)

    stats_dir = Path(args.output_dir) if args.output_dir else Path("checkpoints")
    ckpt_paths = {
        m: checkpoint_path(m, args.model_size, stats_dir, variant=args.variant, embed_dim=args.embed_dim)
        for m in ("mae", "ijepa")
    }

    for p in [era5_path, pangu_path, graphcast_path, ckpt_paths["mae"], ckpt_paths["ijepa"]]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("Building time-aligned index pairs...")
    print("  Pangu:")
    era5_p_idx_all, pangu_idx_all = build_paired_indices(era5_path, pangu_path)
    print("  GraphCast:")
    era5_g_idx_all, gc_idx_all    = build_paired_indices(era5_path, graphcast_path)

    era5_p_idx, pangu_idx = cap_pairs(era5_p_idx_all, pangu_idx_all, args.n_samples, rng)
    era5_g_idx, gc_idx    = cap_pairs(era5_g_idx_all, gc_idx_all,    args.n_samples, rng)
    n_pangu = len(pangu_idx)
    n_gc    = len(gc_idx)
    print(f"\nUsing {n_pangu} Pangu pairs and {n_gc} GraphCast pairs.")

    stats = (
        np.load(stats_dir / "data_mean.npy"),
        np.load(stats_dir / "data_std.npy"),
    )

    print("\nLoading datasets (lazy=False, using ERA5 normalization stats)...")
    era5_ds  = AtmosphereDataset(era5_path,      split="all", stats=stats, lazy=False)
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

    for model_name in ["mae", "ijepa"]:
        print(f"\n{'='*60}")
        print(f"Model: {MODEL_LABELS[model_name]}{label_suffix}")
        print(f"{'='*60}")

        model = build_model(model_name, device, args.model_size, embed_dim=args.embed_dim)
        ckpt  = ckpt_paths[model_name]
        print(f"Loading checkpoint: {ckpt}")
        model = load_model_checkpoint(model_name, model, ckpt, device)
        model.eval()

        # Subsets
        loaders = {
            "era5_p":    DataLoader(Subset(era5_ds,  era5_p_idx), **loader_kw),
            "pangu":     DataLoader(Subset(pangu_ds, pangu_idx),  **loader_kw),
            "era5_g":    DataLoader(Subset(era5_ds,  era5_g_idx), **loader_kw),
            "graphcast": DataLoader(Subset(gc_ds,    gc_idx),     **loader_kw),
        }

        feats = {}
        for name, loader in loaders.items():
            print(f"  Extracting features: {name} ({len(loader.dataset)} samples)...")
            feats[name] = extract_features_for_loader(model, loader, device)

        results[model_name] = {
            "pangu":     compute_distances(feats["era5_p"], feats["pangu"]),
            "graphcast": compute_distances(feats["era5_g"], feats["graphcast"]),
        }
        all_feats[model_name] = {
            "era5":      feats["era5_p"].numpy(),   # representative ERA5 distribution
            "pangu":     feats["pangu"].numpy(),
            "graphcast": feats["graphcast"].numpy(),
        }

    print_metrics_table(results, n_pangu, n_gc, device, label_suffix=label_suffix)

    tag_parts = [args.model_size]
    if args.embed_dim is not None:
        tag_parts.append(f"d{args.embed_dim}")
    if args.variant:
        tag_parts.append(args.variant)
    tag_parts.append(f"n{n_pangu}_seed{args.seed}")
    tag_parts.append(f"pool-{os.environ.get('EXTRACT_FEATURES_POOLING', 'mean').lower()}")
    run_tag = "_".join(tag_parts)

    if args.output_dir:
        plots_dir = Path(args.output_dir) / "plots" / "real_vs_forecast"
    else:
        plots_dir = Path("plots/real_vs_forecast")
    plot_metric_bars(results, plots_dir, run_tag)
    plot_pca_scatter(all_feats, results, plots_dir, run_tag, model_label_suffix=label_suffix)


if __name__ == "__main__":
    main()
