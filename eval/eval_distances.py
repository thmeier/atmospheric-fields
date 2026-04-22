import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from functools import partial
import torch
import numpy as np
import scipy.linalg
from torch.utils.data import DataLoader, Subset

from utils.dataset import AtmosphereDataset
from utils.model_io import build_model, checkpoint_path, load_model_checkpoint
from utils.corruptions import (
    apply_gaussian_blur,
    apply_high_freq_noise,
    apply_gaussian_field_noise,
    apply_random_pixel_replace,
    apply_wind_patch_shuffle,
    apply_wind_channel_rotation,
    get_corruption_ladder,
)

CLUSTER_DATA_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local.nc"
LARGE_LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local_5y.nc"

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"Warning: Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def mmd_rbf(X, Y, gamma=None):
    """
    Computes MMD with RBF kernel between two sets of samples
    gamma corresponds to 1/(2*sigma^2). If None, uses median heuristic.
    X, Y: tensors of shape (N, D)
    """
    XX = torch.cdist(X, X, p=2)**2
    YY = torch.cdist(Y, Y, p=2)**2
    XY = torch.cdist(X, Y, p=2)**2

    if gamma is None:
        # Median heuristic (exclude diagonal self-distances of 0)
        diag_mask = ~torch.eye(XX.shape[0], dtype=torch.bool)
        dists = torch.cat([XX[diag_mask], YY[diag_mask], XY.flatten()])
        gamma = 1.0 / (2.0 * torch.median(dists).item() + 1e-6)

    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)

    # Unbiased MMD statistic
    N, M = X.shape[0], Y.shape[0]

    # Remove diagonal for unbiased XX and YY (self-similarity)
    K_XX.fill_diagonal_(0)
    K_YY.fill_diagonal_(0)

    mmd2 = (K_XX.sum() / (N * (N - 1))) + (K_YY.sum() / (M * (M - 1))) - (2 * K_XY.mean())
    return mmd2.item()


def evaluate_model(model_name, model_size, device, stats_dir, dataset, batch_size, num_workers, local_flags, n_severity_steps=5):
    print(f"Loading {model_name.upper()} model...")
    model = build_model(model_name, device=device, model_size=model_size)
    ckpt_path = checkpoint_path(model_name, model_size, stats_dir)
    model = load_model_checkpoint(model_name, model, ckpt_path, device)
    model.eval()

    ladders = {
        "Gaussian Blur": (get_corruption_ladder("blur", n_severity_steps), apply_gaussian_blur),
        "High-Freq Noise": (get_corruption_ladder("noise", n_severity_steps), apply_high_freq_noise),
        "GRF Noise": (get_corruption_ladder("grf", n_severity_steps), apply_gaussian_field_noise),
        "Random Pixel Replace": (get_corruption_ladder("pixel_replace", n_severity_steps), apply_random_pixel_replace),
        "Spatial Shuffle (Wind Only)": (get_corruption_ladder("wind_patch_shuffle", n_severity_steps), partial(apply_wind_patch_shuffle, patch_size=model.patch_size)),
        "Channel Rotation": (get_corruption_ladder("wind_rotation", n_severity_steps), apply_wind_channel_rotation),
    }

    n_samples = 200 if local_flags["local"] else (400 if local_flags["large_local"] else 1000)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices.tolist())
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    subset_loader = DataLoader(subset, **loader_kwargs)

    print(f"\n--- Validation Protocol 2: Fréchet & MMD Distances ({model_name.upper()}) ---")
    print(f"Reference distribution built from {len(indices)} samples.")

    z_ref = []
    with torch.no_grad():
        for img in subset_loader:
            img = img.to(device, non_blocking=device.type == "cuda")
            z_ref.append(model.extract_features(img).cpu())

    z_ref = torch.cat(z_ref, dim=0)
    z_ref_np = z_ref.numpy()
    mu_ref = np.mean(z_ref_np, axis=0)
    sigma_ref = np.cov(z_ref_np, rowvar=False)

    results = {}
    for cond_name, (severities, apply_fn) in ladders.items():
        print(f"\n--- Corruption: {cond_name} ---")
        results[cond_name] = {"severities": severities, "fid": [], "mmd": []}
        for sev in severities:
            z_cor = []
            with torch.no_grad():
                for img in subset_loader:
                    img = img.to(device, non_blocking=device.type == "cuda")
                    z_cor.append(model.extract_features(apply_fn(img, sev)).cpu())

            z_cor = torch.cat(z_cor, dim=0)
            z_cor_np = z_cor.numpy()
            mu_cor = np.mean(z_cor_np, axis=0)
            sigma_cor = np.cov(z_cor_np, rowvar=False)
            fid = calculate_frechet_distance(mu_ref, sigma_ref, mu_cor, sigma_cor)
            mmd = mmd_rbf(z_ref, z_cor)

            print(f"  Severity {sev} | FID: {fid:8.2f} | MMD: {mmd:8.5f}")
            results[cond_name]["fid"].append(fid)
            results[cond_name]["mmd"].append(mmd)

    return results, ladders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mae", "ijepa", "both"], default="mae")
    parser.add_argument("--model-size", choices=["tiny", "small", "twin", "default"], default="tiny")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--large-local", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lazy", dest="lazy", action="store_true", help="Enable lazy dataloading")
    parser.add_argument("--eager", dest="lazy", action="store_false", help="Force eager dataloading")
    parser.add_argument("--n-severity-steps", type=int, default=9)
    parser.set_defaults(lazy=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available() and device.type != 'cuda':
        device = torch.device('cpu')

    if args.local and args.large_local:
        raise ValueError("Use only one of --local or --large-local.")

    if args.local:
        data_path = LOCAL_DATA_PATH
    elif args.large_local:
        data_path = LARGE_LOCAL_DATA_PATH
    else:
        data_path = CLUSTER_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print("Loading data stats...")
    stats_dir = Path("checkpoints")
    mean = np.load(stats_dir / "data_mean.npy")
    std = np.load(stats_dir / "data_std.npy")
    stats = (mean, std)

    lazy_load = (not args.local) if args.lazy is None else args.lazy
    num_workers = 0 if args.num_workers is None else args.num_workers
    dataset = AtmosphereDataset(data_path, split="val", stats=stats, lazy=lazy_load)

    models_to_run = ["mae", "ijepa"] if args.model == "both" else [args.model]
    all_results = {}
    ladders = None
    for model_name in models_to_run:
        results, ladders = evaluate_model(
            model_name=model_name,
            model_size=args.model_size,
            device=device,
            stats_dir=stats_dir,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            local_flags={"local": args.local, "large_local": args.large_local},
            n_severity_steps=args.n_severity_steps,
        )
        all_results[model_name] = results

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plots_dir = Path("plots") if (args.local or args.large_local) else Path("/work/scratch/ddemler/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "Gaussian Blur": "blue",
        "High-Freq Noise": "red",
        "GRF Noise": "green",
        "Random Pixel Replace": "orange",
        "Spatial Shuffle (Wind Only)": "purple",
        "Channel Rotation": "brown",
    }
    markers = {
        "Gaussian Blur": "o",
        "High-Freq Noise": "s",
        "GRF Noise": "^",
        "Random Pixel Replace": "D",
        "Spatial Shuffle (Wind Only)": "P",
        "Channel Rotation": "X",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    model_styles = {"mae": "-", "ijepa": "-."}
    model_widths = {"mae": 2.0, "ijepa": 3.0}
    model_alphas = {"mae": 0.8, "ijepa": 1.0}
    for model_name, results in all_results.items():
        for cond_name in ladders.keys():
            sevs = results[cond_name]["severities"]
            fids = results[cond_name]["fid"]
            mmds = results[cond_name]["mmd"]
            label = cond_name if args.model != "both" else None
            ax1.plot(
                sevs,
                fids,
                label=label,
                marker=markers[cond_name],
                color=colors[cond_name],
                linewidth=model_widths[model_name],
                linestyle=model_styles[model_name],
                alpha=model_alphas[model_name],
            )
            ax2.plot(
                sevs,
                mmds,
                label=label,
                marker=markers[cond_name],
                color=colors[cond_name],
                linewidth=model_widths[model_name],
                linestyle=model_styles[model_name],
                alpha=model_alphas[model_name],
            )

    title_model = "MAE vs IJEPA" if args.model == "both" else args.model.upper()
    ax1.set_title(f"Fréchet Distance vs. Corruption Severity ({title_model})")
    ax1.set_xlabel("Severity Level")
    ax1.set_ylabel("Fréchet Distance")
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f"Maximum Mean Discrepancy (MMD) vs. Corruption Severity ({title_model})")
    ax2.set_xlabel("Severity Level")
    ax2.set_ylabel("MMD")
    ax2.set_yscale("symlog", linthresh=1e-3)
    ax2.grid(True, alpha=0.3)

    if args.model == "both":
        corruption_handles = [
            Line2D([0], [0], color=colors[name], marker=markers[name], linewidth=2, linestyle="-", label=name)
            for name in ladders.keys()
        ]
        model_handles = [
            Line2D(
                [0], [0],
                color="black",
                linewidth=2.5,
                linestyle=model_styles["mae"],
                marker=None,
                label="MAE (solid)",
            ),
            Line2D(
                [0], [0],
                color="black",
                linewidth=2.5,
                linestyle=model_styles["ijepa"],
                marker=None,
                label="IJEPA (dash-dot)",
            ),
        ]
        legend1 = ax1.legend(handles=corruption_handles, title="Corruption", loc="upper left")
        ax1.add_artist(legend1)
        ax1.legend(handles=model_handles, title="Model", loc="upper right")

        legend2 = ax2.legend(handles=corruption_handles, title="Corruption", loc="upper left")
        ax2.add_artist(legend2)
        ax2.legend(handles=model_handles, title="Model", loc="upper right")
    else:
        ax1.legend()
        ax2.legend()

    plot_path = plots_dir / f"distances_vs_severity_{args.model}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved distance plots to {plot_path}")

    # ── Normalized per-corruption subplots (FID and MMD as separate figures) ──
    corruption_names = list(ladders.keys())
    n_corruptions = len(corruption_names)
    ncols = 3
    nrows = (n_corruptions + ncols - 1) // ncols

    model_colors = {"mae": "#2196F3", "ijepa": "#FF5722"}
    model_labels = {"mae": "MAE", "ijepa": "I-JEPA"}

    for metric_key, metric_label in [("fid", "FID"), ("mmd", "MMD")]:
        fig2, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for ax_idx, cond_name in enumerate(corruption_names):
            ax = axes[ax_idx]
            for model_name, results in all_results.items():
                sevs = np.array(results[cond_name]["severities"])
                vals = np.array(results[cond_name][metric_key])
                vals_norm = (vals - vals[0]) / (np.abs(vals[0]) + 1e-8)

                color = model_colors.get(model_name, "gray")
                label = model_labels.get(model_name, model_name)
                ax.plot(sevs, vals_norm, color=color, linestyle="-", marker="o", markersize=4, label=label)

            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.set_title(cond_name, fontsize=11)
            ax.set_xlabel("Severity")
            ax.set_ylabel(f"Normalised {metric_label} increase")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for ax_idx in range(n_corruptions, len(axes)):
            axes[ax_idx].set_visible(False)

        fig2.suptitle(
            f"Normalised {metric_label} Increase vs. Corruption Severity ({title_model})\n"
            r"$(d - d_0)\,/\,|d_0|$",
            fontsize=13,
            y=1.01,
        )
        norm_plot_path = plots_dir / f"distances_normalised_{metric_key}_{args.model}.png"
        fig2.tight_layout()
        fig2.savefig(norm_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved normalised {metric_label} plots to {norm_plot_path}")

    # ── Summary table ──
    print("\n" + "=" * 80)
    print("DISTANCE EVALUATION SUMMARY")
    print("=" * 80)
    for model_name, results in all_results.items():
        print(f"\nModel: {model_name.upper()}")
        header = f"  {'Corruption':<30} {'Sev':>5} {'FID':>10} {'MMD':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cond_name, data in results.items():
            for sev, fid, mmd in zip(data["severities"], data["fid"], data["mmd"]):
                print(f"  {cond_name:<30} {sev:5.2f} {fid:10.2f} {mmd:10.5f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
