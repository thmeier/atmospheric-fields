"""K-fold lead-time analysis for neural-model holdout discriminators.

Each neural forecast model is evaluated by the discriminator that was trained
without that model.  Numerical baselines are evaluated against the same
discriminator and averaged for a compact comparison.
"""

import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch

try:
    from .analysis_utils import (
        LeadTimeInferenceDataset,
        mean_logits_by_lead,
        normalization_stats,
        resolve_device,
    )
    from .train_discriminator import WeatherDiscriminator, safe_open_dataset, select_time_ranges
except ImportError:
    from analysis_utils import (
        LeadTimeInferenceDataset,
        mean_logits_by_lead,
        normalization_stats,
        resolve_device,
    )
    from train_discriminator import WeatherDiscriminator, safe_open_dataset, select_time_ranges


NUMERICAL_MODELS = {"IFS HRES", "ERA5 Forecast"}


def safe_model_name(name):
    """Convert a comparison label into the training-script filename tag."""
    return name.replace(" ", "_").replace("/", "_")


def load_comparison_datasets(cfg, model_vars, means, stds):
    """Open comparison files and split them into neural and numerical groups."""
    ml_datasets = {}
    numerical_datasets = {}
    max_lead_hour = 0

    for label, path in cfg.comparison_files.items():
        if not os.path.exists(path):
            print(f"Skipping {label}: file not found at {path}")
            continue

        ds = safe_open_dataset(path)
        if cfg.selected_variable not in ds.data_vars:
            print(f"Skipping {label}: variable {cfg.selected_variable} not found.")
            ds.close()
            continue

        test_ds = ds.sel(time=slice(cfg.test_fake_range[0], cfg.test_fake_range[1]))
        if test_ds.sizes.get("time", 0) == 0:
            print(f"Skipping {label}: no samples in test range.")
            ds.close()
            continue

        dataset = LeadTimeInferenceDataset(test_ds, model_vars, means, stds, level=cfg.get("level"))
        max_lead_hour = max(max_lead_hour, int(dataset.lead_hours.max()))

        if label in NUMERICAL_MODELS:
            numerical_datasets[label] = dataset
        else:
            ml_datasets[label] = dataset

    return ml_datasets, numerical_datasets, max_lead_hour


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Plot holdout and numerical lead-time curves for k-fold discriminators."""
    device = resolve_device()
    model_vars = [cfg.selected_variable]

    real_ds = safe_open_dataset(cfg.test_real_nc_file)
    real_test_ds = select_time_ranges(real_ds, cfg.test_real_ranges)
    means, stds = normalization_stats(real_ds, model_vars, cfg.train_real_range)

    model = WeatherDiscriminator(len(model_vars), cfg.model_name).to(device)
    era5_dataset = LeadTimeInferenceDataset(real_test_ds, model_vars, means, stds, level=cfg.get("level"))
    ml_datasets, numerical_datasets, max_lead_hour = load_comparison_datasets(cfg, model_vars, means, stds)

    plot_results = []
    for ml_label, ml_dataset in ml_datasets.items():
        checkpoint = Path(cfg.output_dir) / (
            f"discriminator_{cfg.model_name}_{cfg.selected_variable}_exclude_{safe_model_name(ml_label)}.pth"
        )
        if not checkpoint.exists():
            print(f"Skipping {ml_label}: holdout checkpoint not found at {checkpoint}")
            continue

        print(f"Evaluating {ml_label} with checkpoint: {checkpoint}")
        model.model.load_state_dict(torch.load(checkpoint, map_location=device))

        lead_hours, ml_mean, ml_std = mean_logits_by_lead(ml_dataset, model, cfg, device, f"Holdout -> {ml_label}")
        _, era5_mean, era5_std = mean_logits_by_lead(era5_dataset, model, cfg, device, f"ERA5 on {ml_label}-holdout")

        numerical_means = []
        numerical_stds = []
        for num_label, num_dataset in numerical_datasets.items():
            _, num_mean, num_std = mean_logits_by_lead(
                num_dataset,
                model,
                cfg,
                device,
                f"Num ({num_label}) on {ml_label}-holdout",
            )
            numerical_means.append(num_mean)
            numerical_stds.append(num_std)

        if numerical_means:
            avg_num_mean = np.mean(numerical_means, axis=0)
            avg_num_std = np.mean(numerical_stds, axis=0)
        else:
            avg_num_mean = np.full_like(np.asarray(ml_mean), np.nan, dtype=float)
            avg_num_std = np.full_like(np.asarray(ml_std), np.nan, dtype=float)

        plot_results.append(
            {
                "label": ml_label,
                "lead_hours": lead_hours,
                "ml_mean": ml_mean,
                "ml_std": ml_std,
                "num_mean": avg_num_mean,
                "num_std": avg_num_std,
                "era5_mean": era5_mean[0],
                "era5_std": era5_std[0],
            }
        )

    if not plot_results:
        raise RuntimeError("No k-fold results were produced. Check comparison files and checkpoint names.")

    plt.figure(figsize=(15, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_results)))

    avg_era5_mean = float(np.mean([r["era5_mean"] for r in plot_results]))
    avg_era5_std = float(np.mean([r["era5_std"] for r in plot_results]))
    plt.axhline(avg_era5_mean, color="black", linewidth=1.5, label="ERA5 (Reference Mean)", zorder=10)
    plt.fill_between(
        [-5, max_lead_hour + 5],
        avg_era5_mean - avg_era5_std,
        avg_era5_mean + avg_era5_std,
        color="black",
        alpha=0.1,
        label="ERA5 +/- 1 sigma",
        zorder=1,
    )
    plt.errorbar([0], [avg_era5_mean], yerr=[avg_era5_std], fmt="o", color="black", markersize=8, capsize=10, zorder=11)

    for color, result in zip(colors, plot_results):
        order = np.argsort(result["lead_hours"])
        lead_hours = result["lead_hours"][order]
        ml_mean = np.asarray(result["ml_mean"])[order]
        ml_std = np.asarray(result["ml_std"])[order]
        num_mean = np.asarray(result["num_mean"])[order]
        num_std = np.asarray(result["num_std"])[order]

        plt.errorbar(
            lead_hours,
            ml_mean,
            yerr=ml_std,
            fmt="-o",
            color=color,
            linewidth=2,
            capsize=4,
            alpha=0.9,
            label=f"ML: {result['label']} (Holdout)",
        )
        if not np.isnan(num_mean).all():
            plt.errorbar(
                lead_hours,
                num_mean,
                yerr=num_std,
                fmt="--s",
                color=color,
                linewidth=1.5,
                capsize=3,
                alpha=0.6,
                label=f"Numerical Avg (on {result['label']} Discr)",
            )

    plt.axhline(0, color="black", linestyle="-", alpha=0.3)
    plt.xlim(-5, max_lead_hour + 5)
    plt.xlabel("Lead Time (hours)", fontsize=12)
    plt.ylabel("Discriminator Logit Output", fontsize=12)
    plt.text(0.02, 0.96, "REAL-LIKE (Logits > 0)", color="green", fontweight="bold", transform=plt.gca().transAxes)
    plt.text(0.02, 0.04, "FAKE-LIKE (Logits < 0)", color="red", fontweight="bold", transform=plt.gca().transAxes)

    var_display = cfg.selected_variable.replace("_", " ").title()
    plt.title(f"K-Fold Holdout Analysis: Neural vs Numerical Generalization\nModel: {cfg.model_name} | Variable: {var_display}", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.7)

    output_path = os.path.join(cfg.output_dir, f"kfold_comparison_ML_vs_Numerical_{cfg.model_name}_{cfg.selected_variable}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    real_ds.close()
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
