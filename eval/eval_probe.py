import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from functools import partial
import torch
import numpy as np
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
    MAX_SEVERITY,
)

CLUSTER_DATA_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local.nc"
LARGE_LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local_5y.nc"


def evaluate_model(model_name, ijepa_size, device, stats_dir, dataset, batch_size, num_workers, local_flags, n_probe_samples=None):
    print(f"Loading {model_name.upper()} model...")
    model = build_model(model_name, device=device, ijepa_size=ijepa_size)
    ckpt_path = checkpoint_path(model_name, stats_dir)
    model = load_model_checkpoint(model_name, model, ckpt_path, device)
    model.eval()

    corruption_fns = {
        "Gaussian Blur": apply_gaussian_blur,
        "High-Freq Noise": apply_high_freq_noise,
        "GRF Noise": apply_gaussian_field_noise,
        "Random Pixel Replace": apply_random_pixel_replace,
        "Spatial Shuffle (Wind Only)": partial(apply_wind_patch_shuffle, patch_size=model.patch_size),
        "Channel Rotation": apply_wind_channel_rotation,
    }

    print(f"\n--- Validation Protocol 1: Linear Regression Probe (Continuous Severity Regression, {model_name.upper()}) ---")

    default_n = 50 if local_flags["local"] else (250 if local_flags["large_local"] else 1000)
    n_samples = n_probe_samples if n_probe_samples is not None else default_n
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

    results = {}
    scatters = {}
    for corr_name, apply_fn in corruption_fns.items():
        print(f"\n--- Corruption: {corr_name} ---")
        x_z = []
        y_severity = []

        print(f"Extracting latents for {len(indices)} samples with independent random severities...")
        with torch.no_grad():
            for base_img in subset_loader:
                base_img = base_img.to(device, non_blocking=device.type == "cuda")
                batch_len = base_img.shape[0]
                for j in range(batch_len):
                    sev = np.random.uniform(0.0, MAX_SEVERITY)
                    corrupted = apply_fn(base_img[j:j + 1], severity=sev)
                    x_z.append(model.extract_features(corrupted).cpu())
                    y_severity.append(sev)

        x_z = torch.cat(x_z, dim=0)
        y_severity = torch.tensor(y_severity, dtype=torch.float32).unsqueeze(1)
        print(f"Extracted feature matrix: {x_z.shape}")

        split_idx = max(1, int(0.8 * x_z.shape[0]))
        perm = torch.randperm(x_z.shape[0])
        train_idx = perm[:split_idx]
        test_idx = perm[split_idx:]
        if test_idx.numel() == 0:
            test_idx = train_idx

        x_train = x_z[train_idx]
        y_train = y_severity[train_idx]
        x_test = x_z[test_idx]
        y_test = y_severity[test_idx]

        probe = torch.nn.Linear(x_train.shape[1], 1).to(device)

        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        probe_batch_size = min(64, x_train.shape[0])
        x_train_d = x_train.to(device)
        y_train_d = y_train.to(device)
        x_test_d = x_test.to(device)

        probe.train()
        for epoch in range(200):
            perm_epoch = torch.randperm(x_train_d.shape[0], device=device)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, x_train_d.shape[0], probe_batch_size):
                idx = perm_epoch[i:i + probe_batch_size]
                pred = probe(x_train_d[idx])
                loss = torch.nn.functional.mse_loss(pred, y_train_d[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 50 == 0:
                print(f"  Probe epoch {epoch+1}/200, train MSE: {epoch_loss/n_batches:.4f}")

        probe.eval()
        with torch.no_grad():
            y_pred = probe(x_test_d).cpu()
        y_true = y_test.cpu()

        mse = torch.mean((y_pred - y_true) ** 2).item()
        y_var = torch.var(y_true).item()
        r2 = 1 - mse / y_var if y_var > 0 else 0.0
        print(f"\nLinear Probe Results ({corr_name}, continuous severity):")
        print(f"  MSE: {mse:.4f}")
        print(f"  R^2: {r2:.4f}")

        results[corr_name] = {"mse": mse, "r2": r2}
        scatters[corr_name] = {
            "y_true": y_true.numpy().flatten(),
            "y_pred": y_pred.numpy().flatten(),
        }

    return results, scatters, corruption_fns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mae", "ijepa", "both"], default="mae")
    parser.add_argument("--ijepa-size", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--large-local", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lazy", dest="lazy", action="store_true", help="Enable lazy dataloading")
    parser.add_argument("--eager", dest="lazy", action="store_false", help="Force eager dataloading")
    parser.set_defaults(lazy=None)
    parser.add_argument("--n-probe-samples", type=int, default=None)
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

    import matplotlib.pyplot as plt
    plots_dir = Path("plots") if (args.local or args.large_local) else Path("/work/scratch/ddemler/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_to_run = ["mae", "ijepa"] if args.model == "both" else [args.model]
    all_results = {}
    all_scatters = {}
    corruption_fns = None
    for model_name in models_to_run:
        results, scatters, corruption_fns = evaluate_model(
            model_name=model_name,
            ijepa_size=args.ijepa_size,
            device=device,
            stats_dir=stats_dir,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            local_flags={"local": args.local, "large_local": args.large_local},
            n_probe_samples=args.n_probe_samples,
        )
        all_results[model_name] = results
        all_scatters[model_name] = scatters

    if args.model == "both":
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        corr_names = list(corruption_fns.keys())
        x = np.arange(len(corr_names))
        width = 0.35
        mae_r2 = [all_results["mae"][name]["r2"] for name in corr_names]
        ijepa_r2 = [all_results["ijepa"][name]["r2"] for name in corr_names]
        axes[0].bar(x - width / 2, mae_r2, width=width, label="MAE", color="steelblue")
        axes[0].bar(x + width / 2, ijepa_r2, width=width, label="IJEPA", color="darkorange")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(corr_names, rotation=30, ha="right")
        axes[0].set_ylabel("$R^2$")
        axes[0].set_title("Linear Probe Severity Prediction (MAE vs IJEPA)")
        axes[0].legend()
        axes[0].grid(True, axis="y", alpha=0.3)

        improvement = np.array(ijepa_r2) - np.array(mae_r2)
        axes[1].bar(x, improvement, color=["darkorange" if val >= 0 else "firebrick" for val in improvement])
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(corr_names, rotation=30, ha="right")
        axes[1].set_ylabel("IJEPA $R^2$ - MAE $R^2$")
        axes[1].set_title("Probe Comparison Delta by Corruption")
        axes[1].grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        plot_path = plots_dir / "probe_comparison_both.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparison plot to {plot_path}")

        fig, axes = plt.subplots(len(models_to_run), len(corruption_fns), figsize=(7 * len(corruption_fns), 6 * len(models_to_run)))
        if len(models_to_run) == 1:
            axes = np.expand_dims(axes, axis=0)
        if len(corruption_fns) == 1:
            axes = np.expand_dims(axes, axis=1)

        for row_idx, model_name in enumerate(models_to_run):
            for col_idx, corr_name in enumerate(corruption_fns.keys()):
                ax = axes[row_idx, col_idx]
                y_true_np = all_scatters[model_name][corr_name]["y_true"]
                y_pred_np = all_scatters[model_name][corr_name]["y_pred"]
                r2 = all_results[model_name][corr_name]["r2"]

                ax.scatter(y_true_np, y_pred_np, alpha=0.3, s=10, color="steelblue")
                lims = [0, MAX_SEVERITY]
                ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')

                fit = np.polyfit(y_true_np, y_pred_np, 1)
                fit_x = np.linspace(0, MAX_SEVERITY, 100)
                ax.plot(fit_x, np.polyval(fit, fit_x), 'r-', linewidth=2, label=f'Best fit (slope={fit[0]:.2f})')

                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.set_title(f"{model_name.upper()} | {corr_name}\n$R^2$ = {r2:.4f} (linear probe)")
                ax.set_xlabel("True Severity")
                ax.set_ylabel("Predicted Severity")
                ax.grid(True, alpha=0.3)
                if col_idx == 0:
                    ax.legend(loc="upper left")

        fig.tight_layout()
        scatter_path = plots_dir / "probe_scatter_combined_both.png"
        fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined scatter plot to {scatter_path}")

        # ── Summary table ──
        print("\n" + "=" * 80)
        print("PROBE EVALUATION SUMMARY")
        print("=" * 80)
        header = f"  {'Corruption':<30} {'MAE R²':>10} {'IJEPA R²':>10} {'Delta':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for corr_name in corruption_fns.keys():
            mae_r2 = all_results["mae"][corr_name]["r2"]
            ij_r2 = all_results["ijepa"][corr_name]["r2"]
            print(f"  {corr_name:<30} {mae_r2:10.4f} {ij_r2:10.4f} {ij_r2 - mae_r2:+10.4f}")
        print("=" * 80)
    else:
        fig, axes = plt.subplots(1, len(corruption_fns), figsize=(7 * len(corruption_fns), 6))
        if len(corruption_fns) == 1:
            axes = [axes]

        model_name = models_to_run[0]
        for ax, corr_name in zip(axes, corruption_fns.keys()):
            y_true_np = all_scatters[model_name][corr_name]["y_true"]
            y_pred_np = all_scatters[model_name][corr_name]["y_pred"]
            r2 = all_results[model_name][corr_name]["r2"]

            ax.scatter(y_true_np, y_pred_np, alpha=0.3, s=10, color="steelblue")
            lims = [0, MAX_SEVERITY]
            ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')

            fit = np.polyfit(y_true_np, y_pred_np, 1)
            fit_x = np.linspace(0, MAX_SEVERITY, 100)
            ax.plot(fit_x, np.polyval(fit, fit_x), 'r-', linewidth=2, label=f'Best fit (slope={fit[0]:.2f})')

            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(f"Linear Probe ({model_name.upper()}): {corr_name}\n$R^2$ = {r2:.4f}")
            ax.set_xlabel("True Severity")
            ax.set_ylabel("Predicted Severity")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = plots_dir / f"probe_scatter_combined_{args.model}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined scatter plot to {plot_path}")

        # ── Summary table ──
        print("\n" + "=" * 80)
        print("PROBE EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nModel: {model_name.upper()}")
        header = f"  {'Corruption':<30} {'R²':>10} {'MSE':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for corr_name in corruption_fns.keys():
            r2 = all_results[model_name][corr_name]["r2"]
            mse = all_results[model_name][corr_name]["mse"]
            print(f"  {corr_name:<30} {r2:10.4f} {mse:10.4f}")
        print("=" * 80)

if __name__ == "__main__":
    main()
