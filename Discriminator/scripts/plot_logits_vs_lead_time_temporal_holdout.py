"""Lead-time plot for per-model temporal holdout discriminators."""

import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch

try:
    from .plot_bundles import categorical_colors, model_colors
    from .analysis_utils import resolve_device
    from .monthly_split import concatenate_forecasts, forecast_pairs, lead_hours
    from .train_target_discriminator_baselines import logits_for, matched_statistics
    from .temporal_holdout_utils import checkpoint_path, discover_temporal_pairs, safe_model_name, variable_tag, variables_from_config
    from .train_discriminator import WeatherDiscriminator, normalize_prediction_timedelta, safe_open_dataset
except ImportError:
    from plot_bundles import categorical_colors, model_colors
    from analysis_utils import resolve_device
    from monthly_split import concatenate_forecasts, forecast_pairs, lead_hours
    from train_target_discriminator_baselines import logits_for, matched_statistics
    from temporal_holdout_utils import checkpoint_path, discover_temporal_pairs, safe_model_name, variable_tag, variables_from_config
    from train_discriminator import WeatherDiscriminator, normalize_prediction_timedelta, safe_open_dataset


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
    model = WeatherDiscriminator(len(model_vars), cfg.model_name).to(device)

    plt.figure(figsize=(13, 8))
    colors = model_colors(pairs)
    era5_means = []
    era5_stds = []
    max_lead_hour = 0

    for color, (model_label, files) in zip(colors, pairs.items()):
        ckpt = resolve_checkpoint(cfg, model_label)
        if not ckpt.exists():
            print(f"Skipping {model_label}: checkpoint not found at {ckpt}")
            continue
        print(f"Evaluating {model_label} with checkpoint: {ckpt}")
        model.model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        opened = [normalize_prediction_timedelta(safe_open_dataset(path)) for path in files["files"]]
        fake_ds = concatenate_forecasts(opened)
        train_pairs = forecast_pairs(fake_ds, real_ds, cfg, "train", cfg.lead_times)
        test_pairs = forecast_pairs(fake_ds, real_ds, cfg, "test", cfg.lead_times)
        if not train_pairs or not test_pairs:
            print(f"Skipping {model_label}: no exact monthly train/test forecast–ERA5 pairs")
            fake_ds.close()
            continue
        means, stds = matched_statistics(real_ds, train_pairs, model_vars)
        curve_hours, fake_mean, fake_std, reference_values = [], [], [], []
        for lead_index, lead_hour in enumerate(lead_hours(fake_ds)):
            selected = [pair for pair in test_pairs if pair.lead_index == lead_index]
            if not selected:
                continue
            forecast_logits = logits_for(
                model, fake_ds, model_vars, means, stds, device,
                int(cfg.get("max_samples", 0)), int(cfg.batch_size),
                lead=lead_index, selected_indices=[pair.forecast_index for pair in selected],
            )
            era5_logits = logits_for(
                model, real_ds, model_vars, means, stds, device,
                int(cfg.get("max_samples", 0)), int(cfg.batch_size),
                selected_indices=[pair.era5_index for pair in selected],
            )
            curve_hours.append(int(lead_hour))
            fake_mean.append(float(np.mean(forecast_logits)))
            fake_std.append(float(np.std(forecast_logits)))
            reference_values.extend(era5_logits.tolist())
        curve_hours = np.asarray(curve_hours)
        max_lead_hour = max(max_lead_hour, int(np.max(curve_hours)))
        plot_curve(model_label, curve_hours, fake_mean, fake_std, color)
        era5_means.append(float(np.mean(reference_values)))
        era5_stds.append(float(np.std(reference_values)))
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
        f"Monthly Valid-Time Holdout: days 1–15 train, days 20–26 test\n"
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
