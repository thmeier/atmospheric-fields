"""Bootstrap null-distribution for latent FID/MMD on ERA5 self-splits.

Phase 1: N_TRIALS random 50/50 splits of ERA5; reports mean±std FID/MMD.
Phase 2 (optional): overlays ERA5 vs Pangu / GraphCast on the same axes.

Usage (local smoke test):
    python eval/eval_era5_self_distance.py --local --model mae --n-trials 3 --n-per-split 20

Usage (cluster, both phases):
    python eval/eval_era5_self_distance.py --n-trials 20 --n-per-split 250 \\
        --pangu-path /cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/pangu_surface_2020_lead24h.nc \\
        --graphcast-path /cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/graphcast_surface_2020_lead24h.nc
"""

import os
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
# Paths
# ---------------------------------------------------------------------------

LOCAL_ERA5_PATH        = Path("data/test_data_local.nc")
LARGE_LOCAL_ERA5_PATH  = Path("data/test_data_local_5y.nc")
CLUSTER_ERA5_PATH      = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
CLUSTER_PANGU_PATH     = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/pangu_surface_2020_lead24h.nc")
CLUSTER_GRAPHCAST_PATH = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/graphcast_surface_2020_lead24h.nc")

# ---------------------------------------------------------------------------
# Visual style (consistent with eval_real_vs_forecast.py)
# ---------------------------------------------------------------------------

MODEL_COLORS  = {"mae": "#2196F3", "ijepa": "#FF5722"}
MODEL_LABELS  = {"mae": "MAE (twin)", "ijepa": "I-JEPA (twin)"}
SOURCE_COLORS = {"era5": "#4CAF50", "pangu": "#9C27B0", "graphcast": "#FF9800"}
SOURCE_LABELS = {"era5": "ERA5", "pangu": "Pangu-24h", "graphcast": "GraphCast-24h"}

# ---------------------------------------------------------------------------
# Metric functions (copied from eval_real_vs_forecast.py — self-contained)
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


def compute_distances(z_a: torch.Tensor, z_b: torch.Tensor) -> dict:
    """FID + MMD between two (N, D) CPU tensors."""
    a, b = z_a.numpy(), z_b.numpy()
    mu_a, sig_a = a.mean(0), np.cov(a, rowvar=False)
    mu_b, sig_b = b.mean(0), np.cov(b, rowvar=False)
    return {
        "fid": calculate_frechet_distance(mu_a, sig_a, mu_b, sig_b),
        "mmd": mmd_rbf(z_a, z_b),
    }

# ---------------------------------------------------------------------------
# Time-alignment helpers (copied from eval_real_vs_forecast.py)
# ---------------------------------------------------------------------------

def _read_times(nc_path):
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path, decode_times=True)
        raw = ds.time.values
        ds.close()
        if hasattr(raw[0], "strftime"):
            return pd.DatetimeIndex([pd.Timestamp(str(t)) for t in raw])
        return pd.DatetimeIndex(raw)
    except Exception:
        from netCDF4 import Dataset as NC4, num2date
        with NC4(nc_path) as ds:
            tvar = ds.variables["time"]
            cftimes = num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"))
        return pd.DatetimeIndex([
            pd.Timestamp(t.year, t.month, t.day, t.hour, t.minute, t.second)
            for t in cftimes
        ])


def build_paired_indices(era5_path, forecast_path, max_dt_hours=6):
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
    total = len(era5_idx)
    if n >= total:
        return era5_idx, fc_idx
    perm = rng.permutation(total)[:n].tolist()
    return [era5_idx[i] for i in perm], [fc_idx[i] for i in perm]

# ---------------------------------------------------------------------------
# Phase 1: Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_era5_self(
    z_all: torch.Tensor,
    n_per_split: int,
    n_trials: int,
    rng: np.random.Generator,
) -> dict:
    """Repeat random 50/50 ERA5 splits and compute FID+MMD each time.

    Returns {"fid": list[float], "mmd": list[float]} of length n_trials.
    """
    N = len(z_all)
    if N < 2 * n_per_split:
        raise ValueError(
            f"Need at least {2 * n_per_split} ERA5 samples for n_per_split={n_per_split}, "
            f"but only {N} available. Reduce --n-per-split."
        )
    fid_list, mmd_list = [], []
    for trial in range(n_trials):
        perm = rng.permutation(N)
        z_a = z_all[perm[:n_per_split]]
        z_b = z_all[perm[n_per_split : 2 * n_per_split]]
        d = compute_distances(z_a, z_b)
        fid_list.append(d["fid"])
        mmd_list.append(d["mmd"])
        print(f"    Trial {trial + 1:2d}/{n_trials} | FID: {d['fid']:8.3f} | MMD: {d['mmd']:10.6f}")
    return {"fid": fid_list, "mmd": mmd_list}

# ---------------------------------------------------------------------------
# Phase 2: Forecast comparison
# ---------------------------------------------------------------------------

def extract_forecast_features(
    model,
    era5_ds: AtmosphereDataset,
    forecast_ds: AtmosphereDataset,
    era5_path: Path,
    forecast_path: Path,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple:
    """Return (z_era5_paired, z_forecast) tensors of shape (n_paired, D)."""
    era5_idx_all, fc_idx_all = build_paired_indices(era5_path, forecast_path)
    era5_idx, fc_idx = cap_pairs(era5_idx_all, fc_idx_all, n_samples, rng)
    loader_kw = dict(batch_size=batch_size, shuffle=False, num_workers=0,
                     pin_memory=device.type == "cuda")
    z_era5 = extract_features_for_loader(
        model, DataLoader(Subset(era5_ds, era5_idx), **loader_kw), device)
    z_fc = extract_features_for_loader(
        model, DataLoader(Subset(forecast_ds, fc_idx), **loader_kw), device)
    return z_era5, z_fc

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results_table(bootstrap_results, forecast_results, models_to_run, n_trials, n_per_split, device):
    hdr = "=" * 88
    print(f"\n{hdr}")
    print("ERA5 SELF-DISTANCE — BOOTSTRAP NULL DISTRIBUTION")
    print(hdr)
    print(f"Device: {device}    N trials: {n_trials}    N per split: {n_per_split}")
    print()
    col = f"  {'Model':<24} {'Comparison':<18} {'FID':>18} {'MMD':>18}"
    print(col)
    print("  " + "-" * (len(col) - 2))
    for model_name in models_to_run:
        label = MODEL_LABELS[model_name]
        br = bootstrap_results[model_name]
        fid_m, fid_s = np.mean(br["fid"]), np.std(br["fid"])
        mmd_m, mmd_s = np.mean(br["mmd"]), np.std(br["mmd"])
        print(f"  {label:<24} {'ERA5 ↔ ERA5':<18} {fid_m:>8.3f} ± {fid_s:<7.3f} {mmd_m:>8.6f} ± {mmd_s:<8.6f}")
        if model_name in forecast_results:
            for source, d in forecast_results[model_name].items():
                src_label = SOURCE_LABELS[source]
                print(f"  {label:<24} {f'ERA5 ↔ {src_label}':<18} {d['fid']:>18.3f} {d['mmd']:>18.6f}")
        print()
    print(hdr)


def plot_bootstrap_distribution(bootstrap_results, models_to_run, plots_dir, run_tag):
    fig, (ax_fid, ax_mmd) = plt.subplots(1, 2, figsize=(12, 5))
    x_positions = list(range(len(models_to_run)))
    rng_jitter = np.random.default_rng(42)

    for ax, metric_key, metric_label in [
        (ax_fid, "fid", "Fréchet Distance"),
        (ax_mmd, "mmd", "MMD"),
    ]:
        all_vals = [np.array(bootstrap_results[m][metric_key]) for m in models_to_run]
        y_max = max(v.max() for v in all_vals)

        for xi, model_name, vals in zip(x_positions, models_to_run, all_vals):
            color = MODEL_COLORS[model_name]

            parts = ax.violinplot(vals, positions=[xi], showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.45)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                if key in parts:
                    parts[key].set_edgecolor(color)
                    parts[key].set_linewidth(1.5)

            jitter = rng_jitter.uniform(-0.06, 0.06, len(vals))
            ax.scatter(xi + jitter, vals, color=color, alpha=0.75, s=22, zorder=3)

            mean_val, std_val = vals.mean(), vals.std()
            ax.text(xi, y_max * 1.04, f"μ={mean_val:.3f}\n±{std_val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        ax.set_xticks(x_positions)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models_to_run])
        ax.set_ylabel(metric_label)
        ax.set_title(f"ERA5 ↔ ERA5 Bootstrap ({metric_label})")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"ERA5 Self-Distance Bootstrap\nRun: {run_tag}", fontsize=12)
    fig.tight_layout()
    out = plots_dir / f"bootstrap_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_comparison_bars(bootstrap_results, forecast_results, models_to_run,
                         sources_present, plots_dir, run_tag):
    all_sources = ["era5"] + sources_present
    n_bars = len(all_sources)
    width = 0.20
    offsets = np.linspace(-(n_bars - 1) * width / 2, (n_bars - 1) * width / 2, n_bars)

    fig, (ax_fid, ax_mmd) = plt.subplots(1, 2, figsize=(12, 5))
    x_positions = np.arange(len(models_to_run))

    for ax, metric_key, metric_label in [
        (ax_fid, "fid", "Fréchet Distance"),
        (ax_mmd, "mmd", "MMD"),
    ]:
        for si, source in enumerate(all_sources):
            color = SOURCE_COLORS[source]
            add_label = True
            for xi, model_name in zip(x_positions, models_to_run):
                xpos = xi + offsets[si]
                if source == "era5":
                    vals = np.array(bootstrap_results[model_name][metric_key])
                    mean_val, std_val = vals.mean(), vals.std()
                    bar = ax.bar(xpos, mean_val, width, color=color,
                                 label="ERA5 ↔ ERA5 (bootstrap)" if add_label else None,
                                 edgecolor="white", linewidth=0.5)
                    ax.errorbar(xpos, mean_val, yerr=std_val,
                                fmt="none", color="black", capsize=4, linewidth=1.5)
                    ax.text(xpos, mean_val + std_val * 1.05,
                            f"{mean_val:.2f}", ha="center", va="bottom", fontsize=7)
                elif model_name in forecast_results and source in forecast_results[model_name]:
                    val = forecast_results[model_name][source][metric_key]
                    ax.bar(xpos, val, width, color=color,
                           label=SOURCE_LABELS[source] if add_label else None,
                           edgecolor="white", linewidth=0.5)
                    ax.text(xpos, val * 1.02, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=7)
                add_label = False  # only label the first model's bar per source

        ax.set_xticks(x_positions)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models_to_run])
        ax.set_ylabel(metric_label)
        ax.set_title(f"ERA5 ↔ ERA5 vs Forecasts ({metric_label})")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"ERA5 Self-Distance vs Forecast Distance\nRun: {run_tag}", fontsize=12)
    fig.tight_layout()
    out = plots_dir / f"comparison_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap ERA5 self-distance null distribution for FID/MMD.")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--large-local", action="store_true")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-per-split", type=int, default=250,
                        help="Samples per split; need 2×n_per_split ≤ total ERA5 samples.")
    parser.add_argument("--model", choices=["mae", "ijepa", "both"], default="both")
    parser.add_argument("--model-size", choices=["default", "twin", "tiny", "small"], default="twin")
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pangu-path", type=str, default=None)
    parser.add_argument("--graphcast-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.local and args.large_local:
        raise ValueError("Use only one of --local or --large-local.")

    era5_path = (LOCAL_ERA5_PATH if args.local
                 else LARGE_LOCAL_ERA5_PATH if args.large_local
                 else CLUSTER_ERA5_PATH)
    if not era5_path.exists():
        raise FileNotFoundError(f"ERA5 dataset not found: {era5_path}")

    stats_dir = Path(args.output_dir) if args.output_dir else Path("checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and device.type != "cuda":
        device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    stats = (np.load(stats_dir / "data_mean.npy"), np.load(stats_dir / "data_std.npy"))
    lazy_load = not (args.local or args.large_local)
    era5_ds = AtmosphereDataset(era5_path, split="all", stats=stats, lazy=lazy_load)
    print(f"ERA5 dataset: {len(era5_ds)} samples (split='all')")

    # Resolve optional forecast paths
    pangu_path     = Path(args.pangu_path)     if args.pangu_path     else None
    graphcast_path = Path(args.graphcast_path) if args.graphcast_path else None

    phase2_sources = []
    for name, path in [("pangu", pangu_path), ("graphcast", graphcast_path)]:
        if path is not None:
            if path.exists():
                phase2_sources.append(name)
            else:
                print(f"Warning: --{name}-path {path} does not exist; skipping {name}.")

    models_to_run = ["mae", "ijepa"] if args.model == "both" else [args.model]
    bootstrap_results: dict = {}
    forecast_results: dict  = {}

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {MODEL_LABELS[model_name]}")
        print(f"{'='*60}")

        model = build_model(model_name, device, args.model_size, embed_dim=args.embed_dim)
        ckpt  = checkpoint_path(model_name, args.model_size, stats_dir,
                                variant=args.variant, embed_dim=args.embed_dim)
        model = load_model_checkpoint(model_name, model, ckpt, device)
        model.eval()

        # Extract all ERA5 features in one pass
        full_loader = DataLoader(era5_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0,
                                 pin_memory=device.type == "cuda")
        print(f"Extracting ERA5 features ({len(era5_ds)} samples)...")
        z_all = extract_features_for_loader(model, full_loader, device)
        print(f"Feature shape: {z_all.shape}")

        # Phase 1: bootstrap
        print(f"\nBootstrap: {args.n_trials} trials, n_per_split={args.n_per_split}")
        bootstrap_results[model_name] = bootstrap_era5_self(
            z_all, args.n_per_split, args.n_trials, rng)

        # Phase 2: forecast comparison
        if phase2_sources:
            forecast_results[model_name] = {}
            for source in phase2_sources:
                fc_path = pangu_path if source == "pangu" else graphcast_path
                fc_ds   = AtmosphereDataset(fc_path, split="all", stats=stats, lazy=lazy_load)
                print(f"\nPhase 2 — {SOURCE_LABELS[source]} ({len(fc_ds)} samples):")
                z_era5_paired, z_fc = extract_forecast_features(
                    model, era5_ds, fc_ds, era5_path, fc_path,
                    n_samples=2 * args.n_per_split,
                    batch_size=args.batch_size,
                    device=device,
                    rng=rng,
                )
                forecast_results[model_name][source] = compute_distances(z_era5_paired, z_fc)

    print_results_table(bootstrap_results, forecast_results, models_to_run,
                        args.n_trials, args.n_per_split, device)

    tag_parts = [args.model_size]
    if args.embed_dim is not None:
        tag_parts.append(f"d{args.embed_dim}")
    if args.variant:
        tag_parts.append(args.variant)
    tag_parts.append(f"t{args.n_trials}_n{args.n_per_split}_seed{args.seed}")
    tag_parts.append(f"pool-{os.environ.get('EXTRACT_FEATURES_POOLING', 'mean').lower()}")
    run_tag = "_".join(tag_parts)

    if args.output_dir:
        plots_dir = Path(args.output_dir) / "plots" / "era5_self_distance"
    else:
        plots_dir = Path("plots/era5_self_distance")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_bootstrap_distribution(bootstrap_results, models_to_run, plots_dir, run_tag)
    if forecast_results:
        plot_comparison_bars(bootstrap_results, forecast_results,
                             models_to_run, phase2_sources, plots_dir, run_tag)


if __name__ == "__main__":
    main()
