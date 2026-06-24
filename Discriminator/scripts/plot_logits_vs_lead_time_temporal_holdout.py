"""Lead-time plot for per-model temporal holdout discriminators."""

import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch

try:
    from .analysis_utils import LeadTimeInferenceDataset, mean_logits_by_lead, normalization_stats, resolve_device
    from .temporal_holdout_utils import checkpoint_path, discover_temporal_pairs, safe_model_name, variable_tag, variables_from_config
    from .train_discriminator import WeatherDiscriminator, safe_open_dataset, select_time_ranges
except ImportError:
    from analysis_utils import LeadTimeInferenceDataset, mean_logits_by_lead, normalization_stats, resolve_device
    from temporal_holdout_utils import checkpoint_path, discover_temporal_pairs, safe_model_name, variable_tag, variables_from_config
    from train_discriminator import WeatherDiscriminator, safe_open_dataset, select_time_ranges


def plot_curve(label, lead_hours, means, stds, color):
    """Plot one mean/std logit curve with lead hours sorted on the x-axis."""
    order = np.argsort(lead_hours)
    plt.errorbar(
        lead_hours[order],
        np.asarray(means)[order],
        yerr=np.asarray(stds)[order],
        fmt="-o",
        color=color,
        linewidth=2,
        capsize=4,
        alpha=0.9,
        label=label,
    )


def resolve_checkpoint(cfg, model_label):
    """Find a temporal checkpoint, with fallback for legacy selected-variable names."""
    ckpt = checkpoint_path(cfg, model_label)
    if ckpt.exists():
        return ckpt
    if variable_tag(cfg) != cfg.selected_variable:
        legacy = ckpt.with_name(
            f"discriminator_{cfg.model_name}_{cfg.selected_variable}_"
            f"temporal_{safe_model_name(model_label)}.pth"
        )
        if legacy.exists():
            print(f"Using legacy selected-variable checkpoint path: {legacy}")
            return legacy
    return ckpt


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Plot each temporal discriminator against its own forecast model."""
    device = resolve_device()
    model_vars = variables_from_config(cfg)
    pairs = discover_temporal_pairs(cfg)

    if not pairs:
        raise RuntimeError("No temporal train/test forecast pairs found.")

    real_ds = safe_open_dataset(cfg.real_nc_file)
    real_test_ds = select_time_ranges(real_ds, cfg.test_real_ranges)
    means, stds = normalization_stats(real_ds, model_vars, cfg.train_real_range)

    model = WeatherDiscriminator(len(model_vars), cfg.model_name).to(device)
    era5_dataset = LeadTimeInferenceDataset(
        real_test_ds,
        model_vars,
        means,
        stds,
        level=cfg.get("level"),
        max_samples=cfg.get("max_samples", 0),
    )

    plt.figure(figsize=(13, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(pairs)))
    era5_means = []
    era5_stds = []
    max_lead_hour = 0

    for color, (model_label, files) in zip(colors, pairs.items()):
        ckpt = resolve_checkpoint(cfg, model_label)
        if not ckpt.exists():
            print(f"Skipping {model_label}: checkpoint not found at {ckpt}")
            continue
        if not os.path.exists(files["test"]):
            print(f"Skipping {model_label}: test file not found at {files['test']}")
            continue

        print(f"Evaluating {model_label} with checkpoint: {ckpt}")
        model.model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        fake_ds = safe_open_dataset(files["test"])
        fake_test_ds = fake_ds.sel(time=slice(cfg.test_fake_range[0], cfg.test_fake_range[1]))
        if fake_test_ds.sizes.get("time", 0) == 0:
            print(f"Skipping {model_label}: no fake samples in test range {cfg.test_fake_range}")
            fake_ds.close()
            continue

        fake_dataset = LeadTimeInferenceDataset(
            fake_test_ds,
            model_vars,
            means,
            stds,
            level=cfg.get("level"),
            max_samples=cfg.get("max_samples", 0),
        )
        lead_hours, fake_mean, fake_std = mean_logits_by_lead(
            fake_dataset,
            model,
            cfg,
            device,
            f"{model_label} test forecast",
        )
        max_lead_hour = max(max_lead_hour, int(np.max(lead_hours)))
        plot_curve(model_label, lead_hours, fake_mean, fake_std, color)

        _, era5_mean, era5_std = mean_logits_by_lead(
            era5_dataset,
            model,
            cfg,
            device,
            f"ERA5 with {model_label} discriminator",
        )
        era5_means.append(era5_mean[0])
        era5_stds.append(era5_std[0])
        fake_ds.close()

    if not era5_means:
        raise RuntimeError("No temporal holdout results were produced.")

    avg_era5_mean = float(np.mean(era5_means))
    avg_era5_std = float(np.mean(era5_stds))
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

    plt.axhline(0, color="black", linestyle="-", alpha=0.3)
    plt.xlim(-5, max_lead_hour + 5)
    plt.xlabel("Lead Time (hours)", fontsize=12)
    plt.ylabel("Discriminator Logit Output", fontsize=12)
    plt.text(0.02, 0.96, "REAL-LIKE (Logits > 0)", color="green", fontweight="bold", transform=plt.gca().transAxes)
    plt.text(0.02, 0.04, "FAKE-LIKE (Logits < 0)", color="red", fontweight="bold", transform=plt.gca().transAxes)

    var_display = cfg.selected_variable.replace("_", " ").title() if len(model_vars) == 1 else "All Fields"
    plt.title(
        f"Temporal Holdout Analysis: Train Forecast Period vs Test Forecast Period\n"
        f"Model: {cfg.model_name} | Variable: {var_display}",
        fontsize=14,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.7)

    output_path = os.path.join(
        cfg.output_dir,
        f"temporal_holdout_comparison_{cfg.model_name}_{variable_tag(cfg)}.png",
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    real_ds.close()
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
