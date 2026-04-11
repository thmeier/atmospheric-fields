import argparse
from pathlib import Path
from functools import partial
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from dataset import AtmosphereDataset
from models import MaskedAutoencoderViT
from corruptions import (
    apply_gaussian_blur,
    apply_high_freq_noise,
    apply_gaussian_field_noise,
    apply_random_pixel_replace,
    apply_wind_patch_shuffle,
    apply_wind_channel_rotation,
    MAX_SEVERITY,
)

CLUSTER_DATA_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
LOCAL_DATA_PATH = Path(__file__).parent / "data" / "test_data_local.nc"
LARGE_LOCAL_DATA_PATH = Path(__file__).parent / "data" / "test_data_local_5y.nc"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--large-local", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lazy", dest="lazy", action="store_true", help="Enable lazy dataloading")
    parser.add_argument("--eager", dest="lazy", action="store_false", help="Force eager dataloading")
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

    print("Loading MAE model...")
    model = MaskedAutoencoderViT(
        embed_dim=256, depth=6, num_heads=8,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4
    ).to(device)

    ckpt_path = stats_dir / "best_mae_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    corruption_fns = {
        "Gaussian Blur": apply_gaussian_blur,
        "High-Freq Noise": apply_high_freq_noise,
        "GRF Noise": apply_gaussian_field_noise,
        "Random Pixel Replace": apply_random_pixel_replace,
        "Spatial Shuffle (Wind Only)": partial(apply_wind_patch_shuffle, patch_size=model.patch_size),
        "Channel Rotation": apply_wind_channel_rotation,
    }

    print("\n--- Validation Protocol 1: Linear Probe (Continuous Severity Regression) ---")

    n_samples = 50 if args.local else (250 if args.large_local else 1000)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices.tolist())
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    subset_loader = DataLoader(subset, **loader_kwargs)

    import matplotlib.pyplot as plt
    plots_dir = Path("plots") if (args.local or args.large_local) else Path("/work/scratch/ddemler/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(corruption_fns), figsize=(7 * len(corruption_fns), 6))
    if len(corruption_fns) == 1:
        axes = [axes]

    for ax, (corr_name, apply_fn) in zip(axes, corruption_fns.items()):
        print(f"\n--- Corruption: {corr_name} ---")

        X_z = []
        y_severity = []

        print(f"Extracting latents for {len(indices)} samples with independent random severities...")

        with torch.no_grad():
            for base_img in subset_loader:
                base_img = base_img.to(device, non_blocking=device.type == "cuda")
                B = base_img.shape[0]
                for j in range(B):
                    sev = np.random.uniform(0.0, MAX_SEVERITY)
                    single = base_img[j:j+1]
                    corrupted = apply_fn(single, severity=sev)
                    z = model.extract_features(corrupted)
                    X_z.append(z.cpu())
                    y_severity.append(sev)

        X_z = torch.cat(X_z, dim=0)
        y_severity = torch.tensor(y_severity, dtype=torch.float32).unsqueeze(1)

        print(f"Extracted feature matrix: {X_z.shape}")

        split_idx = max(1, int(0.8 * X_z.shape[0]))
        perm = torch.randperm(X_z.shape[0])
        train_idx = perm[:split_idx]
        test_idx = perm[split_idx:]
        if test_idx.numel() == 0:
            test_idx = train_idx

        X_train = X_z[train_idx]
        y_train = y_severity[train_idx]
        X_test = X_z[test_idx]
        y_test = y_severity[test_idx]

        hidden_dim = 128
        probe = torch.nn.Sequential(
            torch.nn.Linear(X_train.shape[1], hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        ).to(device)

        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        num_epochs = 200
        probe_batch_size = min(64, X_train.shape[0])

        X_train_d = X_train.to(device)
        y_train_d = y_train.to(device)
        X_test_d = X_test.to(device)
        y_test_d = y_test.to(device)

        probe.train()
        for epoch in range(num_epochs):
            perm_epoch = torch.randperm(X_train_d.shape[0], device=device)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, X_train_d.shape[0], probe_batch_size):
                idx = perm_epoch[i:i+probe_batch_size]
                pred = probe(X_train_d[idx])
                loss = torch.nn.functional.mse_loss(pred, y_train_d[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 50 == 0:
                print(f"  Probe epoch {epoch+1}/{num_epochs}, train MSE: {epoch_loss/n_batches:.4f}")

        probe.eval()
        with torch.no_grad():
            y_pred = probe(X_test_d).cpu()
        y_test = y_test.cpu()

        mse = torch.mean((y_pred - y_test)**2).item()
        y_var = torch.var(y_test).item()
        r2 = 1 - mse / y_var if y_var > 0 else 0.0

        print(f"\nMLP Probe Results ({corr_name}, continuous severity):")
        print(f"  MSE: {mse:.4f}")
        print(f"  R^2: {r2:.4f}")

        if r2 > 0.5:
            print("-> SUCCESS: The latent space encodes corruption severity.")
        else:
            print("-> FAILURE: The probe failed to predict severity.")

        y_true_np = y_test.numpy().flatten()
        y_pred_np = y_pred.numpy().flatten()

        ax.scatter(y_true_np, y_pred_np, alpha=0.3, s=10, color="steelblue")
        lims = [0, MAX_SEVERITY]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')

        fit = np.polyfit(y_true_np, y_pred_np, 1)
        fit_x = np.linspace(0, MAX_SEVERITY, 100)
        ax.plot(fit_x, np.polyval(fit, fit_x), 'r-', linewidth=2,
                label=f'Best fit (slope={fit[0]:.2f})')

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"MLP Probe: {corr_name}\n$R^2$ = {r2:.4f}")
        ax.set_xlabel("True Severity")
        ax.set_ylabel("Predicted Severity")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = plots_dir / "probe_scatter_combined.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined scatter plot to {plot_path}")

if __name__ == "__main__":
    main()
