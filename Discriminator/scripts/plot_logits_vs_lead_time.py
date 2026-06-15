"""Plot discriminator logits as a function of forecast lead time.

Positive logits indicate fields the discriminator considers ERA5-like; negative
logits indicate forecast/corruption-like fields.  This script evaluates one
trained discriminator against all files listed in `comparison_files`.
"""

import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import xarray as xr

try:
    from .analysis_utils import (
        LeadTimeInferenceDataset,
        load_discriminator,
        mean_logits_by_lead,
        normalization_stats,
        resolve_device,
    )
    from .train_discriminator import select_time_ranges
except ImportError:
    from analysis_utils import (
        LeadTimeInferenceDataset,
        load_discriminator,
        mean_logits_by_lead,
        normalization_stats,
        resolve_device,
    )
    from train_discriminator import select_time_ranges


def plot_curve(label, lead_hours, means, stds):
    """Plot one mean/std logit curve with lead hours sorted on the x-axis."""
    order = np.argsort(lead_hours)
    plt.errorbar(
        lead_hours[order],
        [means[i] for i in order],
        yerr=[stds[i] for i in order],
        fmt="-o",
        label=label,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Plot one discriminator's mean logits over lead time for comparison files."""
    device = resolve_device()
    model_vars = [cfg.selected_variable]

    real_ds = xr.open_dataset(cfg.test_real_nc_file)
    real_test_ds = select_time_ranges(real_ds, cfg.test_real_ranges)
    means, stds = normalization_stats(real_ds, model_vars, cfg.train_real_range)

    model, model_path = load_discriminator(cfg, model_vars, device)
    print(f"Loaded discriminator: {model_path}")

    plt.figure(figsize=(10, 6))

    # ERA5 has no useful forecast lead dimension here, so it is plotted at zero.
    era5_dataset = LeadTimeInferenceDataset(real_test_ds, model_vars, means, stds, level=cfg.get("level"))
    _, era5_mean, era5_std = mean_logits_by_lead(era5_dataset, model, cfg, device, "Baseline (ERA5)")
    plt.errorbar(
        [0],
        [era5_mean[0]],
        yerr=[era5_std[0]],
        fmt="o",
        label="ERA5 (Ground Truth)",
        markersize=10,
        color="black",
    )

    for label, path in cfg.comparison_files.items():
        if not os.path.exists(path):
            print(f"Skipping {label}: file not found at {path}")
            continue

        fake_ds = xr.open_dataset(path)
        if cfg.selected_variable not in fake_ds.data_vars:
            print(f"Skipping {label}: variable {cfg.selected_variable} not found.")
            fake_ds.close()
            continue

        fake_test_ds = fake_ds.sel(time=slice(cfg.test_fake_range[0], cfg.test_fake_range[1]))
        if fake_test_ds.sizes.get("time", 0) == 0:
            print(f"Skipping {label}: no samples in test range.")
            fake_ds.close()
            continue

        dataset = LeadTimeInferenceDataset(fake_test_ds, model_vars, means, stds, level=cfg.get("level"))
        lead_hours, mean_logits, std_logits = mean_logits_by_lead(dataset, model, cfg, device, label)
        plot_curve(label, lead_hours, mean_logits, std_logits)
        fake_ds.close()

    plt.axhline(0, color="black", linestyle="-", alpha=0.3)
    plt.xlabel("Lead Time (hours)", fontsize=12)
    plt.ylabel("Discriminator Logit Output", fontsize=12)
    plt.text(0.02, 0.95, "REAL-LIKE (Logits > 0)", color="green", fontweight="bold", transform=plt.gca().transAxes)
    plt.text(0.02, 0.05, "FAKE-LIKE (Logits < 0)", color="red", fontweight="bold", transform=plt.gca().transAxes)

    var_display = cfg.selected_variable.replace("_", " ").title()
    plt.title(f"Discriminator Logits vs Lead Time\nModel: {cfg.model_name} | Variable: {var_display}", fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.7)

    output_path = os.path.join(cfg.output_dir, f"comparison_logits_vs_lead_time_{cfg.model_name}_{cfg.selected_variable}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    real_ds.close()
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
