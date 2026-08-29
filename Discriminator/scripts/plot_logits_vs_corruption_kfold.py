"""Plot leave-one-corruption-group-out discriminator sensitivity curves."""

import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

try:
    from .plot_bundles import categorical_colors
    from .corruptions import get_corruption_ladder
    from .train_corruption_kfold import (
        corruption_group_tag,
        corruption_kfold_checkpoint_dir,
        corruption_kfold_holdouts,
        corruption_kfold_types,
    )
    from .train_discriminator import (
        WeatherDiscriminator,
        apply_configured_corruption,
        safe_open_dataset,
        select_time_ranges,
    )
    from .analysis_utils import normalization_stats, resolve_device
    from .train_kfold import variable_tag
except ImportError:
    from plot_bundles import categorical_colors
    from corruptions import get_corruption_ladder
    from train_corruption_kfold import (
        corruption_group_tag,
        corruption_kfold_checkpoint_dir,
        corruption_kfold_holdouts,
        corruption_kfold_types,
    )
    from train_discriminator import (
        WeatherDiscriminator,
        apply_configured_corruption,
        safe_open_dataset,
        select_time_ranges,
    )
    from analysis_utils import normalization_stats, resolve_device
    from train_kfold import variable_tag


DISPLAY_NAMES = {
    "blur": "Gaussian Blur",
    "gaussian_blur": "Gaussian Blur",
    "grf": "GRF Noise",
    "gaussian_field_noise": "GRF Noise",
    "hf_noise": "High-Frequency Noise",
    "high_freq_noise": "High-Frequency Noise",
    "pixel_replace": "Pixel Replace",
    "random_pixel_replace": "Pixel Replace",
    "wind_patch_shuffle": "Wind Patch Shuffle",
    "wind_shuffled": "Wind Patch Shuffle",
    "wind_rotation": "Wind Rotation",
    "wind_rotated": "Wind Rotation",
}

LADDER_ALIASES = {
    "gaussian_blur": "blur",
    "hf_noise": "noise",
    "high_freq_noise": "noise",
    "gaussian_field_noise": "grf",
    "random_pixel_replace": "pixel_replace",
    "wind_shuffled": "wind_patch_shuffle",
    "wind_rotated": "wind_rotation",
}


class CorruptedEra5Dataset(Dataset):
    """Apply one configured tensor corruption to ERA5 samples for inference."""

    def __init__(self, ds, variables, means, stds, corruption_type, severity, level=None, max_samples=100):
        self.ds = ds
        if level is not None:
            if "level" in self.ds.dims:
                self.ds = self.ds.sel(level=level)
            elif "pressure_level" in self.ds.dims:
                self.ds = self.ds.sel(pressure_level=level)

        if "prediction_timedelta" in self.ds.dims:
            self.ds = self.ds.isel(prediction_timedelta=0)

        if max_samples > 0 and self.ds.sizes.get("time", 0) > max_samples:
            indices = np.linspace(0, self.ds.sizes["time"] - 1, max_samples, dtype=int)
            self.ds = self.ds.isel(time=indices)

        self.variables = variables
        self.means = means
        self.stds = stds
        self.corruption_type = corruption_type
        self.severity = float(severity)

    def __len__(self):
        return self.ds.sizes.get("time", 0)

    def __getitem__(self, idx):
        ds_slice = self.ds.isel(time=idx)
        channels = []

        for variable in self.variables:
            if variable in ds_slice.data_vars:
                raw = ds_slice[variable].values.astype(np.float32)
                scale = self.stds[variable] if self.stds[variable] > 1e-8 else 1.0
                channels.append(np.nan_to_num((raw - self.means[variable]) / scale, nan=0.0))
            else:
                ref = ds_slice.temperature if "temperature" in ds_slice.data_vars else list(ds_slice.data_vars.values())[0]
                channels.append(np.zeros(ref.shape, dtype=np.float32))

        sample = torch.tensor(np.stack(channels), dtype=torch.float32)
        return apply_configured_corruption(sample, self.corruption_type, self.severity)


def checkpoint_for_holdout_group(cfg, heldout_corruptions):
    """Return the checkpoint path for one held-out corruption group."""
    filename = (
        f"discriminator_{cfg.model_name}_{variable_tag(cfg)}_"
        f"corruption_exclude_{corruption_group_tag(heldout_corruptions)}.pth"
    )
    return Path(corruption_kfold_checkpoint_dir(cfg)) / filename


def corruption_levels(corruption_type, cfg):
    """Return severity levels for one corruption family."""
    n_steps = int(cfg.get("corruption_kfold_plot_steps", 7))
    ladder_name = LADDER_ALIASES.get(corruption_type, corruption_type)
    return get_corruption_ladder(ladder_name, n_steps=n_steps)


def mean_logits(dataset, model, cfg, device):
    """Score a corrupted ERA5 dataset and return mean/std logits."""
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    logits = []
    model.eval()
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs.to(device)).cpu().numpy().flatten()
            logits.extend(outputs)
    return float(np.mean(logits)), float(np.std(logits))


def evaluate(cfg, real_test_ds, means, stds, model_vars, device):
    """Evaluate all corruption-fold checkpoints over all corruption strengths."""
    model = WeatherDiscriminator(len(model_vars), cfg.model_name).to(device)
    max_samples = int(cfg.get("corruption_kfold_eval_samples", 100))
    corruption_types = corruption_kfold_types(cfg)
    results = {}

    for heldout_corruptions in corruption_kfold_holdouts(cfg):
        checkpoint = checkpoint_for_holdout_group(cfg, heldout_corruptions)
        heldout_label = ", ".join(heldout_corruptions)
        heldout_key = corruption_group_tag(heldout_corruptions)
        if not checkpoint.exists():
            print(f"Skipping {heldout_label}: checkpoint not found at {checkpoint}")
            continue

        print(f"Evaluating held-out corruption fold: {heldout_label}")
        model.model.load_state_dict(torch.load(checkpoint, map_location=device))
        results[heldout_key] = {"heldout_corruptions": heldout_corruptions, "curves": {}}

        for corruption_type in corruption_types:
            levels = corruption_levels(corruption_type, cfg)
            means_by_level = []
            stds_by_level = []
            for severity in levels:
                dataset = CorruptedEra5Dataset(
                    real_test_ds,
                    model_vars,
                    means,
                    stds,
                    corruption_type,
                    severity,
                    level=cfg.get("level"),
                    max_samples=max_samples,
                )
                mean, std = mean_logits(dataset, model, cfg, device)
                means_by_level.append(mean)
                stds_by_level.append(std)
            results[heldout_key]["curves"][corruption_type] = {
                "levels": levels,
                "mean": means_by_level,
                "std": stds_by_level,
            }

    return results


def should_plot_curve(view, heldout_corruptions, corruption_type):
    """Filter curves for all/trained-only/heldout-only plot variants."""
    if view == "all":
        return True
    is_holdout_curve = corruption_type in heldout_corruptions
    if view == "trained":
        return not is_holdout_curve
    if view == "heldout":
        return is_holdout_curve
    raise ValueError(f"Unknown plot view: {view}")


def plot_results(cfg, results, view):
    """Create one plot variant for the requested curve subset."""
    corruption_types = corruption_kfold_types(cfg)
    n_cols = 2
    n_rows = int(np.ceil(len(corruption_types) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    colors = categorical_colors(max(len(results), 1))
    for axis_idx, corruption_type in enumerate(corruption_types):
        ax = axes[axis_idx]
        for color, (_, fold_result) in zip(colors, results.items()):
            heldout_corruptions = fold_result["heldout_corruptions"]
            if not should_plot_curve(view, heldout_corruptions, corruption_type):
                continue
            curve = fold_result["curves"][corruption_type]
            is_holdout_curve = corruption_type in heldout_corruptions
            linestyle = "--" if is_holdout_curve else "-"
            label = f"held out {', '.join(heldout_corruptions)}"
            ax.errorbar(
                curve["levels"],
                curve["mean"],
                yerr=curve["std"],
                fmt="o",
                linestyle=linestyle,
                color=color,
                capsize=3,
                linewidth=1.8,
                alpha=0.8,
                label=label,
            )

        ax.axhline(0, color="black", linestyle=":", alpha=0.5)
        ax.set_title(DISPLAY_NAMES.get(corruption_type, corruption_type), fontsize=13, fontweight="bold")
        ax.set_xlabel("Corruption severity")
        ax.set_ylabel("Discriminator logit")
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.94, "REAL-LIKE", transform=ax.transAxes, color="green", fontweight="bold", fontsize=9)
        ax.text(0.02, 0.05, "FAKE-LIKE", transform=ax.transAxes, color="red", fontweight="bold", fontsize=9)

    for ax in axes[len(corruption_types):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.18, 0.5), fontsize=9)

    view_titles = {
        "all": "trained and held-out curves",
        "trained": "trained-on corruption curves only",
        "heldout": "held-out corruption curves only",
    }
    var_display = cfg.selected_variable.replace("_", " ").title() if len(model_vars_from_cfg(cfg)) == 1 else "All Fields"
    fig.suptitle(
        f"Leave-One-Corruption-Out K-Fold: {view_titles[view]}\n"
        f"Model: {cfg.model_name} | Variable: {var_display}",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_path = os.path.join(
        cfg.output_dir,
        f"corruption_kfold_logits_{view}_{cfg.model_name}_{variable_tag(cfg)}.png",
    )
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {view} plot to: {output_path}")


def model_vars_from_cfg(cfg):
    """Return model input fields from config."""
    variables = cfg.get("variables")
    if variables:
        return list(variables)
    return [cfg.selected_variable]


@hydra.main(version_base=None, config_path="../conf", config_name="kfold_config")
def main(cfg: DictConfig):
    """Evaluate leave-one-corruption-out checkpoints on synthetic corruptions."""
    device = resolve_device()
    model_vars = model_vars_from_cfg(cfg)

    real_ds = safe_open_dataset(cfg.get("test_real_nc_file", cfg.real_nc_file))
    real_test_ds = select_time_ranges(real_ds, cfg.get("test_real_ranges", cfg.train_real_range))
    if real_test_ds.sizes.get("time", 0) == 0:
        raise ValueError("No ERA5 samples found in test_real_ranges for corruption k-fold plotting.")

    means, stds = normalization_stats(real_ds, model_vars, cfg.train_real_range)
    results = evaluate(cfg, real_test_ds, means, stds, model_vars, device)
    if not results:
        raise RuntimeError("No corruption k-fold checkpoints were evaluated.")

    for view in ("all", "trained", "heldout"):
        plot_results(cfg, results, view)

    real_ds.close()


if __name__ == "__main__":
    main()
