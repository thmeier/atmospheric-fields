"""Visualize blur, high-freq noise & GRF noise corruption ladders."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import AtmosphereDataset
from corruptions import (
    apply_gaussian_blur,
    apply_high_freq_noise,
    apply_gaussian_field_noise,
    apply_random_pixel_replace,
    get_corruption_ladder,
)

DATA_PATH = Path(__file__).parent / "data" / "test_data_local.nc"
STATS_DIR = Path("checkpoints")

VAR_NAMES = ["2m Temperature", "10m U-Wind", "10m V-Wind", "Mean Sea Level Pressure"]
CMAPS = ["RdBu_r", "coolwarm", "coolwarm", "viridis"]

CORRUPTION_SPECS = [
    ("Gaussian Blur", "blur", apply_gaussian_blur),
    ("High-Freq Noise", "noise", apply_high_freq_noise),
    ("GRF Noise", "grf", apply_gaussian_field_noise),
    ("Random Pixel Replace", "pixel_replace", apply_random_pixel_replace),
]

# Which variables to visualize
SHOW_VARS = [0, 3]  # 2m Temperature and Mean Sea Level Pressure


def main():
    mean = np.load(STATS_DIR / "data_mean.npy")
    std = np.load(STATS_DIR / "data_std.npy")
    stats = (mean, std)

    ds = AtmosphereDataset(DATA_PATH, split="val", stats=stats, lazy=False)
    sample = ds[0].unsqueeze(0)  # (1, 4, H, W)

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Combined figure per variable: 3 rows (corruptions) x N columns (severities)
    for var_idx in SHOW_VARS:
        sevs = get_corruption_ladder("blur")
        n_cols = len(sevs)
        n_rows = len(CORRUPTION_SPECS)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
        sample_var = sample[0, var_idx].numpy()
        vmin = float(sample_var.min())
        vmax = float(sample_var.max())

        for col, sev in enumerate(sevs):
            for row, (corr_name, _, apply_fn) in enumerate(CORRUPTION_SPECS):
                corrupted = apply_fn(sample, sev)
                axes[row, col].imshow(
                    corrupted[0, var_idx].numpy(),
                    cmap=CMAPS[var_idx],
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                if row == 0:
                    axes[row, col].set_title(f"sev={sev:.4f}", fontsize=13)
                axes[row, col].axis("off")

        for row, (corr_name, _, _) in enumerate(CORRUPTION_SPECS):
            axes[row, 0].axis("on")
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            axes[row, 0].set_ylabel(corr_name, fontsize=14, rotation=90, labelpad=10)

        fig.suptitle(f"Corruption Ladders — {VAR_NAMES[var_idx]}", fontsize=16)
        fig.tight_layout()

        safe_var = VAR_NAMES[var_idx].lower().replace(" ", "_").replace("-", "_")
        fname = f"corruption_ladder_combined_{safe_var}.png"
        fig.savefig(plots_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {plots_dir / fname}")


if __name__ == "__main__":
    main()
