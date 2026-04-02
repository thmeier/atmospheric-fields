import argparse
from pathlib import Path
import torch
import numpy as np
import scipy.linalg
from torch.utils.data import DataLoader, Subset

from dataset import AtmosphereDataset
from models import MaskedAutoencoderViT
from corruptions import (
    apply_gaussian_blur,
    apply_high_freq_noise,
    apply_gaussian_field_noise,
    apply_random_pixel_replace,
    get_corruption_ladder,
)

CLUSTER_DATA_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
LOCAL_DATA_PATH = Path(__file__).parent / "data" / "test_data_local.nc"
LARGE_LOCAL_DATA_PATH = Path(__file__).parent / "data" / "test_data_local_5y.nc"

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
    
    # Use smaller sample sets for distances to compute quickly locally
    n_samples = 200 if args.local else (400 if args.large_local else 1000)
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
    
    print(f"\n--- Validation Protocol 2: Fréchet & MMD Distances ---")
    print(f"Reference distribution built from {len(indices)} samples.")
    
    Z_ref = []
    
    with torch.no_grad():
        for img in subset_loader:
            img = img.to(device, non_blocking=device.type == "cuda")
            z = model.extract_features(img)
            Z_ref.append(z.cpu())
            
    Z_ref = torch.cat(Z_ref, dim=0) # (N, embed_dim)
    Z_ref_np = Z_ref.numpy()
    
    mu_ref = np.mean(Z_ref_np, axis=0)
    sigma_ref = np.cov(Z_ref_np, rowvar=False)
    
    ladders = {
        "Gaussian Blur": (get_corruption_ladder("blur"), apply_gaussian_blur),
        "High-Freq Noise": (get_corruption_ladder("noise"), apply_high_freq_noise),
        "GRF Noise": (get_corruption_ladder("grf"), apply_gaussian_field_noise),
        "Random Pixel Replace": (get_corruption_ladder("pixel_replace"), apply_random_pixel_replace),
    }
    
    results = {}
    
    for cond_name, (severities, apply_fn) in ladders.items():
        print(f"\n--- Corridor: {cond_name} ---")
        
        results[cond_name] = {"severities": severities, "fid": [], "mmd": []}
        
        for sev in severities:
            Z_cor = []
            with torch.no_grad():
                for img in subset_loader:
                    img = img.to(device, non_blocking=device.type == "cuda")
                    corrupted = apply_fn(img, sev)
                    z = model.extract_features(corrupted)
                    Z_cor.append(z.cpu())
                    
            Z_cor = torch.cat(Z_cor, dim=0)
            Z_cor_np = Z_cor.numpy()
            
            # Fréchet Distance
            mu_cor = np.mean(Z_cor_np, axis=0)
            sigma_cor = np.cov(Z_cor_np, rowvar=False)
            
            fid = calculate_frechet_distance(mu_ref, sigma_ref, mu_cor, sigma_cor)
            
            # MMD
            mmd = mmd_rbf(Z_ref, Z_cor)
            
            print(f"  Severity {sev} | FID: {fid:8.2f} | MMD: {mmd:8.5f}")
            results[cond_name]["fid"].append(fid)
            results[cond_name]["mmd"].append(mmd)
            
    # Plotting
    import matplotlib.pyplot as plt
    plots_dir = Path("plots") if (args.local or args.large_local) else Path("/work/scratch/ddemler/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        "Gaussian Blur": "blue",
        "High-Freq Noise": "red",
        "GRF Noise": "green",
        "Random Pixel Replace": "orange",
    }
    markers = {
        "Gaussian Blur": "o",
        "High-Freq Noise": "s",
        "GRF Noise": "^",
        "Random Pixel Replace": "D",
    }

    for cond_name in ladders.keys():
        sevs = results[cond_name]["severities"]
        fids = results[cond_name]["fid"]
        mmds = results[cond_name]["mmd"]
        
        ax1.plot(sevs, fids, label=cond_name, marker=markers[cond_name], color=colors[cond_name], linewidth=2)
        ax2.plot(sevs, mmds, label=cond_name, marker=markers[cond_name], color=colors[cond_name], linewidth=2)
        
    ax1.set_title("Fréchet Distance vs. Corruption Severity")
    ax1.set_xlabel("Severity Level")
    ax1.set_ylabel("Fréchet Distance")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title("Maximum Mean Discrepancy (MMD) vs. Corruption Severity")
    ax2.set_xlabel("Severity Level")
    ax2.set_ylabel("MMD")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plot_path = plots_dir / "distances_vs_severity.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved distance plots to {plot_path}")

if __name__ == "__main__":
    main()
