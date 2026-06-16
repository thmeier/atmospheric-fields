"""Real-vs-forecast latent distances using a colleague's pretrained SFNO encoder.

Measures Fréchet (FID-style) distance and RBF-MMD between ERA5 (real) and
GraphCast (24h forecast) surface fields, embedded by the frozen SFNO autoencoder
from the sibling ``SFNO-Embedding`` repo. GraphCast-only by design: SFNO needs a
6h-precipitation channel and Pangu does not forecast precipitation.

This is the SFNO analogue of ``eval_real_vs_forecast.py``. It deliberately reuses
that module's distance + time-alignment helpers so the methodology matches the
MAE/I-JEPA results, and adds:
  * an ERA5-vs-ERA5 baseline (random 50/50 split of the reference pool) — the
    noise floor the forecast distance should be compared against;
  * raw 5-var data loading (SFNO standardizes inputs itself).

Example (local smoke test)::

    /opt/miniconda3/envs/pmlr/bin/python eval/eval_sfno_real_vs_forecast.py \
        --era5-path data/era5_5var_2020-01_local.nc \
        --graphcast-path data/graphcast_5var_2020-01_lead24h_local.nc \
        --channels 10 --res 31 --n-samples 60
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.features import extract_features_for_loader
from utils.sfno_embedding import SFNOEmbedding, RawFiveVarDataset
# Reuse the exact distance + index-alignment logic from the MAE/I-JEPA eval.
from eval.eval_real_vs_forecast import (
    compute_distances,
    mmd_rbf,
    build_era5_ref_pool,
    build_forecast_indices,
    cap_indices,
    split_indices,
)

RES_BY_FIRST = {61: (61, 120), 31: (31, 60), 15: (15, 30)}


def plot_bars(results, plots_dir, run_tag):
    """Bar chart of FID/MMD for ERA5-self baseline vs ERA5-vs-GraphCast.

    Drops the FID panel when FID is NaN (MMD-only runs).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["ERA5 vs ERA5\n(baseline)", "ERA5 vs GraphCast\n(24h)"]
    keys = ["era5_self", "graphcast"]
    panels = [("fid", "Fréchet Distance"), ("mmd", "MMD (RBF)")]
    panels = [(m, t) for m, t in panels
              if not all(results[k][m] != results[k][m] for k in keys)]  # drop all-NaN

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), squeeze=False)
    for ax, (metric, title) in zip(axes[0], panels):
        vals = [results[k][metric] for k in keys]
        bars = ax.bar(labels, vals, color=["#9e9e9e", "#FF5722"])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.4g}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"SFNO embedding — real vs forecast\nRun: {run_tag}", y=1.02)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"sfno_real_vs_forecast_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot → {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era5-path", required=True, type=Path)
    parser.add_argument("--graphcast-path", required=True, type=Path)
    parser.add_argument("--channels", type=int, default=10, choices=[5, 10, 20],
                        help="SFNO embedding channels (default: 10)")
    parser.add_argument("--res", type=int, default=31, choices=[15, 31, 61],
                        help="SFNO embedding resolution (first dim; default: 31 -> 31x60)")
    parser.add_argument("--pooling", choices=["mean", "max", "meanstd", "grid", "flatten"],
                        default="mean",
                        help="How to turn the (C,h,w) SFNO embedding into a feature vector. "
                             "mean/max=C dims, meanstd=2C, grid=C*gh*gw (adaptive-pooled, keeps "
                             "spatial structure), flatten=C*h*w (MMD only — FID is singular). "
                             "Default: mean")
    parser.add_argument("--pool-grid", type=int, nargs=2, default=(7, 8), metavar=("GH", "GW"),
                        help="Target grid for --pooling grid (default: 7 8)")
    parser.add_argument("--mmd-only", action="store_true",
                        help="Compute only MMD (skip FID). Use with high-dim features "
                             "(e.g. --pooling flatten) where FID's covariance is singular.")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Cap on samples per pool (real and forecast)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-dt-hours", type=int, default=6,
                        help="Tolerance for matching ERA5 hours-of-day to forecast valid times")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sfno-repo", type=str, default=None,
                        help="Path to the SFNO-Embedding checkout (else $SFNO_REPO or sibling dir)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write plots/ (default: ./plots)")
    args = parser.parse_args()

    for p in (args.era5_path, args.graphcast_path):
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    res = RES_BY_FIRST[args.res]

    # --- Time-aligned index pools (same logic as the MAE/I-JEPA eval) ---
    print("Building time-aligned index pools (ERA5 ref vs GraphCast)...")
    ref_idx, _ = build_era5_ref_pool(
        args.era5_path, args.graphcast_path,
        max_dt_hours=args.max_dt_hours, prior_hours=None, restrict_date_range=True,
    )
    gc_idx, _ = build_forecast_indices(
        args.era5_path, args.graphcast_path,
        max_dt_hours=args.max_dt_hours, prior_hours=None,
    )
    ref_idx, _ = cap_indices(ref_idx, None, args.n_samples, rng)
    gc_idx, _ = cap_indices(gc_idx, None, args.n_samples, rng)
    # ERA5-self baseline: disjoint 50/50 split of the reference pool.
    (base_a, _), (base_b, _) = split_indices(ref_idx, None, rng)

    n_ref, n_gc, n_base = len(ref_idx), len(gc_idx), len(base_a)
    print(f"\nERA5 ref: {n_ref}    GraphCast: {n_gc}    baseline halves: {n_base} each")

    # --- Model + raw 5-var datasets ---
    print(f"\nLoading SFNO encoder ({args.channels}c, {res[0]}x{res[1]}, pool={args.pooling})...")
    model = SFNOEmbedding(
        embedding_channels=args.channels, embedding_resolution=res,
        repo_root=args.sfno_repo, pooling=args.pooling, pool_grid=tuple(args.pool_grid),
    ).to(device)
    model.eval()
    print(f"  feature_dim = {model.feature_dim}")

    # FID's covariance is singular once feature_dim >= N — warn so the number isn't
    # silently meaningless. MMD stays valid at any dimension.
    fid_valid = model.feature_dim < min(n_ref, n_gc, 2 * n_base)
    if not fid_valid:
        print(f"  WARNING: feature_dim ({model.feature_dim}) >= N — FID is singular/"
              f"unreliable here; trust MMD. Use --pooling grid (smaller --pool-grid) "
              f"or raise --n-samples for a valid FID.")

    era5_ds = RawFiveVarDataset(args.era5_path)
    gc_ds = RawFiveVarDataset(args.graphcast_path)

    loader_kw = dict(batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, pin_memory=device.type == "cuda")

    def feats(ds, idx):
        return extract_features_for_loader(model, DataLoader(Subset(ds, idx), **loader_kw), device)

    print("\nExtracting features...")
    f_ref = feats(era5_ds, ref_idx)
    f_gc = feats(gc_ds, gc_idx)
    f_base_a = feats(era5_ds, base_a)
    f_base_b = feats(era5_ds, base_b)

    if args.mmd_only:
        # Skip FID entirely — at high feature_dim (e.g. flatten) its covariance is
        # singular. MMD only needs pairwise distances and stays valid.
        results = {
            "era5_self": {"fid": float("nan"), "mmd": mmd_rbf(f_base_a, f_base_b)},
            "graphcast": {"fid": float("nan"), "mmd": mmd_rbf(f_ref, f_gc)},
        }
    else:
        results = {
            "era5_self": compute_distances(f_base_a, f_base_b),
            "graphcast": compute_distances(f_ref, f_gc),
        }

    # --- Report ---
    def _fmt(v):
        return "      n/a" if v != v else f"{v:>12.4f}"  # v != v ↔ NaN

    print("\n" + "=" * 70)
    print("SFNO EMBEDDING — REAL vs 24h FORECAST (GraphCast)")
    print("=" * 70)
    print(f"  config: {args.channels}c {res[0]}x{res[1]}  pool={args.pooling}  "
          f"feature_dim={model.feature_dim}  device={device}")
    print(f"  N: ref={n_ref}  graphcast={n_gc}  baseline={n_base} each"
          f"{'   (MMD only)' if args.mmd_only else ''}\n")
    print(f"  {'Comparison':<26} {'FID':>12} {'MMD':>14}")
    print("  " + "-" * 52)
    b, g = results["era5_self"], results["graphcast"]
    print(f"  {'ERA5 vs ERA5 (baseline)':<26} {_fmt(b['fid'])} {b['mmd']:>14.6f}")
    print(f"  {'ERA5 vs GraphCast (24h)':<26} {_fmt(g['fid'])} {g['mmd']:>14.6f}")
    if not args.mmd_only and b['fid']:
        print(f"\n  FID(forecast)/FID(baseline) = {g['fid'] / b['fid']:.2f}  "
              f"(>1 means SFNO separates real from forecast)")
    mmd_ratio = g['mmd'] / b['mmd'] if b['mmd'] else float("nan")
    print(f"  MMD(forecast) - MMD(baseline) = {g['mmd'] - b['mmd']:.6f}  "
          f"(>0 means SFNO separates real from forecast)")
    print("=" * 70)

    run_tag = f"{args.channels}c{res[0]}_pool-{args.pooling}_n{n_gc}_seed{args.seed}"
    plots_dir = (Path(args.output_dir) / "plots" if args.output_dir else Path("plots")) / "sfno_real_vs_forecast"
    plot_bars(results, plots_dir, run_tag)


if __name__ == "__main__":
    main()
