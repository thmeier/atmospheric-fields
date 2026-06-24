"""K-fold lead-time analysis for neural-model holdout discriminators.

Each neural forecast model is evaluated by the discriminator trained without
that model. IFS HRES is the only numerical baseline plotted here; ERA5 Forecast
is excluded because it is not the operational numerical model baseline for this
k-fold comparison.
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


NUMERICAL_BASELINE_LABEL = "IFS HRES"
SKIPPED_NUMERICAL_LABELS = {"ERA5 Forecast"}


def safe_model_name(name):
    """Convert a comparison label into the training-script filename tag."""
    return name.replace(" ", "_").replace("/", "_")


def variables_from_config(cfg):
    """Return all configured input fields, falling back to one selected field."""
    variables = cfg.get("variables")
    if variables:
        return list(variables)
    return [cfg.selected_variable]


def variable_tag(cfg):
    """Return the checkpoint filename tag for configured input channels."""
    variables = variables_from_config(cfg)
    return cfg.selected_variable if len(variables) == 1 else "all_fields"


def kfold_checkpoint_dir(cfg):
    """Return the directory that stores k-fold discriminator weights."""
    return Path(cfg.get("kfold_checkpoint_dir", Path(cfg.output_dir) / "kfold_checkpoints"))


def resolve_kfold_checkpoint(cfg, filename):
    """Find a k-fold checkpoint, with fallback for legacy flat output dirs."""
    checkpoint = kfold_checkpoint_dir(cfg) / filename
    if checkpoint.exists():
        return checkpoint
    legacy_checkpoint = Path(cfg.output_dir) / filename
    if legacy_checkpoint.exists():
        print(f"Using legacy flat checkpoint path: {legacy_checkpoint}")
        return legacy_checkpoint
    return checkpoint


def load_comparison_datasets(cfg, model_vars, means, stds):
    """Open comparison files and split them by how they should be plotted.

    The k-fold plot compares each held-out neural forecast model with one
    numerical reference curve. Keep that reference curve fixed to IFS HRES so
    the plot does not silently average different numerical-style products.
    """
    ml_datasets = {}
    numerical_datasets = {}
    max_lead_hour = 0

    for label, path in cfg.comparison_files.items():
        if not os.path.exists(path):
            print(f"Skipping {label}: file not found at {path}")
            continue

        ds = safe_open_dataset(path)
        missing_vars = [variable for variable in model_vars if variable not in ds.data_vars]
        if missing_vars:
            print(f"Skipping {label}: variables not found: {missing_vars}")
            ds.close()
            continue

        test_ds = select_fake_test_data(ds, cfg, label)

        dataset = LeadTimeInferenceDataset(test_ds, model_vars, means, stds, level=cfg.get("level"))
        max_lead_hour = max(max_lead_hour, int(dataset.lead_hours.max()))

        if label == NUMERICAL_BASELINE_LABEL:
            numerical_datasets[label] = dataset
        elif label in SKIPPED_NUMERICAL_LABELS:
            print(f"Skipping numerical comparison {label}: only {NUMERICAL_BASELINE_LABEL} is plotted.")
        else:
            ml_datasets[label] = dataset

    return ml_datasets, numerical_datasets, max_lead_hour


def configured_real_file(cfg):
    """Return the real/reference file configured for this experiment."""
    return cfg.get("test_real_nc_file", cfg.real_nc_file)


def configured_real_ranges(cfg):
    """Return real ranges with data, falling back when k-fold placeholders are empty."""
    return cfg.get("test_real_ranges", cfg.train_real_range)


def select_fake_test_data(ds, cfg, label):
    """Select fake test data, falling back to all times for k-fold placeholders."""
    test_ds = ds.sel(time=slice(cfg.test_fake_range[0], cfg.test_fake_range[1]))
    if test_ds.sizes.get("time", 0) > 0:
        return test_ds
    print(
        f"Warning: {label} has no samples in test_fake_range={cfg.test_fake_range}; "
        "using all available times for k-fold model-holdout analysis."
    )
    return ds


@hydra.main(version_base=None, config_path="../conf", config_name="kfold_config")
def main(cfg: DictConfig):
    """Plot holdout and numerical lead-time curves for k-fold discriminators."""
    device = resolve_device()
    model_vars = variables_from_config(cfg)

    real_ds = safe_open_dataset(configured_real_file(cfg))
    real_test_ds = select_time_ranges(real_ds, configured_real_ranges(cfg))
    if real_test_ds.sizes.get("time", 0) == 0:
        print("Warning: configured real test ranges are empty; using train_real_range for ERA5 reference.")
        real_test_ds = select_time_ranges(real_ds, cfg.train_real_range)
    means, stds = normalization_stats(real_ds, model_vars, cfg.train_real_range)

    model = WeatherDiscriminator(len(model_vars), cfg.model_name).to(device)
    era5_dataset = LeadTimeInferenceDataset(real_test_ds, model_vars, means, stds, level=cfg.get("level"))
    ml_datasets, numerical_datasets, max_lead_hour = load_comparison_datasets(cfg, model_vars, means, stds)

    plot_results = []
    for ml_label, ml_dataset in ml_datasets.items():
        checkpoint = resolve_kfold_checkpoint(
            cfg,
            f"discriminator_{cfg.model_name}_{variable_tag(cfg)}_exclude_{safe_model_name(ml_label)}.pth",
        )
        if not checkpoint.exists() and variable_tag(cfg) != cfg.selected_variable:
            checkpoint = resolve_kfold_checkpoint(
                cfg,
                f"discriminator_{cfg.model_name}_{cfg.selected_variable}_exclude_{safe_model_name(ml_label)}.pth",
            )
        if not checkpoint.exists():
            print(f"Skipping {ml_label}: holdout checkpoint not found at {checkpoint}")
            continue

        print(f"Evaluating {ml_label} with checkpoint: {checkpoint}")
        model.model.load_state_dict(torch.load(checkpoint, map_location=device))

        lead_hours, ml_mean, ml_std = mean_logits_by_lead(ml_dataset, model, cfg, device, f"Holdout -> {ml_label}")
        _, era5_mean, era5_std = mean_logits_by_lead(era5_dataset, model, cfg, device, f"ERA5 on {ml_label}-holdout")

        if NUMERICAL_BASELINE_LABEL in numerical_datasets:
            _, num_mean, num_std = mean_logits_by_lead(
                numerical_datasets[NUMERICAL_BASELINE_LABEL],
                model,
                cfg,
                device,
                f"{NUMERICAL_BASELINE_LABEL} on {ml_label}-holdout",
            )
        else:
            num_mean = np.full_like(np.asarray(ml_mean), np.nan, dtype=float)
            num_std = np.full_like(np.asarray(ml_std), np.nan, dtype=float)

        plot_results.append(
            {
                "label": ml_label,
                "lead_hours": lead_hours,
                "ml_mean": ml_mean,
                "ml_std": ml_std,
                "num_mean": num_mean,
                "num_std": num_std,
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
                label=f"{NUMERICAL_BASELINE_LABEL} (on {result['label']} Discr)",
            )

    plt.axhline(0, color="black", linestyle="-", alpha=0.3)
    plt.xlim(-5, max_lead_hour + 5)
    plt.xlabel("Lead Time (hours)", fontsize=12)
    plt.ylabel("Discriminator Logit Output", fontsize=12)
    plt.text(0.02, 0.96, "REAL-LIKE (Logits > 0)", color="green", fontweight="bold", transform=plt.gca().transAxes)
    plt.text(0.02, 0.04, "FAKE-LIKE (Logits < 0)", color="red", fontweight="bold", transform=plt.gca().transAxes)

    var_display = cfg.selected_variable.replace("_", " ").title() if len(model_vars) == 1 else "All Fields"
    plt.title(f"K-Fold Holdout Analysis: Neural vs Numerical Generalization\nModel: {cfg.model_name} | Variable: {var_display}", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.7)

    output_path = os.path.join(cfg.output_dir, f"kfold_comparison_ML_vs_Numerical_{cfg.model_name}_{variable_tag(cfg)}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    real_ds.close()
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
