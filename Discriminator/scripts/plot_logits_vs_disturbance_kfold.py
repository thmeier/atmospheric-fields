import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from omegaconf import DictConfig

try:
    from .analysis_utils import normalization_stats
    from .corruptions import get_corruption_ladder
    from .train_discriminator import (
        WeatherDiscriminator,
        apply_configured_corruption,
        safe_open_dataset,
        select_time_ranges,
    )
except ImportError:
    from analysis_utils import normalization_stats
    from corruptions import get_corruption_ladder
    from train_discriminator import (
        WeatherDiscriminator,
        apply_configured_corruption,
        safe_open_dataset,
        select_time_ranges,
    )

class DisturbanceInferenceDataset(Dataset):
    """Apply one disturbance level to ERA5 samples before discriminator scoring."""

    def __init__(self, ds, variables, means, stds, disturbance_type, severity, level=None, max_samples=100):
        self.ds = ds
        if level is not None:
            if 'level' in self.ds.dims: self.ds = self.ds.sel(level=level)
            elif 'pressure_level' in self.ds.dims: self.ds = self.ds.sel(pressure_level=level)
        
        self.variables = variables
        self.means = means
        self.stds = stds
        self.disturbance_type = disturbance_type
        self.severity = severity
        
        if "prediction_timedelta" in self.ds.dims:
            self.ds = self.ds.isel(prediction_timedelta=0)

        all_times = self.ds.time.values
        if max_samples > 0 and len(all_times) > max_samples:
            indices = np.linspace(0, len(all_times) - 1, max_samples, dtype=int)
            self.times = all_times[indices]
            self.ds = self.ds.isel(time=indices)
        else:
            self.times = all_times

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        ds_slice = self.ds.isel(time=idx)
        channels = []
        for v in self.variables:
            if v in ds_slice.data_vars:
                raw_data = ds_slice[v].values.astype(np.float32)
                norm = (raw_data - self.means[v]) / (self.stds[v] if self.stds[v] > 1e-8 else 1.0)
                channels.append(np.nan_to_num(norm, nan=0.0))
            else:
                ref_shape = ds_slice.temperature.shape if 'temperature' in ds_slice.data_vars else list(ds_slice.data_vars.values())[0].shape
                channels.append(np.zeros(ref_shape, dtype=np.float32))

        sample = torch.tensor(np.stack(channels), dtype=torch.float32)
        return apply_configured_corruption(sample, self.disturbance_type, self.severity)

def get_mean_logit(dataset, model, cfg, device):
    """Return mean and standard deviation of discriminator logits."""
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    logits = []
    model.eval()
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs.to(device)).cpu().numpy().flatten()
            logits.extend(outputs)
    return np.mean(logits), np.std(logits)


DISTURBANCE_CATALOG = {
    "gaussian_blur": {"name": "Gaussian Blur", "xlabel": "Severity"},
    "grf": {"name": "GRF Noise", "xlabel": "Severity"},
    "pixel_replace": {"name": "Pixel Replace", "xlabel": "Severity"},
    "wind_patch_shuffle": {"name": "Wind Patch Shuffle", "xlabel": "Severity"},
    "hf_noise": {"name": "HF Noise", "xlabel": "Severity"},
    "wind_rotation": {"name": "Wind Vector Rotation", "xlabel": "Severity"},
}

DISTURBANCE_ALIASES = {
    "blur": "gaussian_blur",
    "noise": "hf_noise",
    "high_freq_noise": "hf_noise",
    "grf_noise": "grf",
    "gaussian_field_noise": "grf",
    "random_pixel_replace": "pixel_replace",
    "wind_shuffled": "wind_patch_shuffle",
    "wind_rotated": "wind_rotation",
}


def canonical_disturbance_name(name):
    """Map training/config names onto the plotting disturbance keys."""
    return DISTURBANCE_ALIASES.get(str(name), str(name))


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


def kfold_checkpoint_files(cfg):
    """List k-fold checkpoints, with fallback for legacy flat output dirs."""
    checkpoint_dir = kfold_checkpoint_dir(cfg)
    if checkpoint_dir.exists():
        files = [
            path for path in checkpoint_dir.iterdir()
            if path.suffix == ".pth" and ("_exclude_" in path.name or "_full_pool" in path.name)
        ]
        if files:
            return files

    legacy_dir = Path(cfg.output_dir)
    files = [
        path for path in legacy_dir.iterdir()
        if path.suffix == ".pth" and ("_exclude_" in path.name or "_full_pool" in path.name)
    ]
    if files:
        print(f"Using legacy flat checkpoint directory: {legacy_dir}")
    return files


def configured_disturbances(cfg):
    """Return disturbances requested by the config, marking held-out probes."""
    training_types = [canonical_disturbance_name(name) for name in cfg.get("corruption_types", [])]
    held_out_types = [canonical_disturbance_name(name) for name in cfg.get("held_out_corruption_types", [])]
    requested = training_types + held_out_types
    if not requested:
        requested = list(DISTURBANCE_CATALOG)

    disturbances = {}
    held_out_set = set(held_out_types)
    for dtype in requested:
        if dtype not in DISTURBANCE_CATALOG or dtype in disturbances:
            continue
        dinfo = dict(DISTURBANCE_CATALOG[dtype])
        dinfo["levels"] = get_corruption_ladder(dtype, n_steps=int(cfg.get("corruption_kfold_plot_steps", 7)))
        dinfo["held_out"] = dtype in held_out_set
        if dinfo["held_out"]:
            dinfo["name"] = f"{dinfo['name']} (Holdout)"
        disturbances[dtype] = dinfo
    return disturbances


def configured_real_file(cfg):
    """Return the real/reference file configured for this experiment."""
    return cfg.get("test_real_nc_file", cfg.real_nc_file)


def configured_real_ranges(cfg):
    """Return real ranges with data, falling back when k-fold placeholders are empty."""
    return cfg.get("test_real_ranges", cfg.train_real_range)


@hydra.main(version_base=None, config_path="../conf", config_name="kfold_config")
def main(cfg: DictConfig):
    """Compare disturbance sensitivity across k-fold/full-pool discriminators."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_vars = variables_from_config(cfg)
    NUMERICAL_MODELS = ["IFS HRES", "ERA5 Forecast"]
    
    real_ds_global = safe_open_dataset(configured_real_file(cfg))
    means, stds = normalization_stats(real_ds_global, model_vars, cfg.train_real_range)
    era5_test_ds = select_time_ranges(real_ds_global, configured_real_ranges(cfg))
    if era5_test_ds.sizes.get("time", 0) == 0:
        print("Warning: configured real test ranges are empty; using train_real_range for ERA5 reference.")
        era5_test_ds = select_time_ranges(real_ds_global, cfg.train_real_range)

    disturbances = configured_disturbances(cfg)
    if not disturbances:
        raise ValueError("No valid disturbances configured for k-fold plotting.")

    model = WeatherDiscriminator(len(model_vars), cfg.model_name).to(device)
    
    # Evaluate only holdout/full-pool discriminator weights produced by
    # `train_kfold.py`.
    model_files = kfold_checkpoint_files(cfg)
    
    results = {} # label -> dtype -> {'m': [], 's': []}

    for model_path in model_files:
        m_file = model_path.name
        if "_exclude_" in m_file:
            label = m_file.split("_exclude_")[1].replace(".pth", "").replace("_", " ")
            if label in NUMERICAL_MODELS or label.replace(" ", "_") in [m.replace(" ", "_") for m in NUMERICAL_MODELS]:
                continue
            label = f"Excl: {label}"
        elif "_full_pool" in m_file:
            label = "Full AI Pool"
        else:
            continue
            
        print(f"Evaluating Discriminator: {label}")
        model.model.load_state_dict(torch.load(model_path, map_location=device))
        results[label] = {}
        
        for dtype, dinfo in disturbances.items():
            results[label][dtype] = {'m': [], 's': []}
            for level in dinfo["levels"]:
                ds_disturbed = DisturbanceInferenceDataset(era5_test_ds, model_vars, means, stds, dtype, level, level=cfg.get("level"), max_samples=50)
                m, s = get_mean_logit(ds_disturbed, model, cfg, device)
                results[label][dtype]['m'].append(m)
                results[label][dtype]['s'].append(s)

    # --- Plotting Line Graphs ---
    n_cols = 2
    n_rows = int(np.ceil(len(disturbances) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (dtype, dinfo) in enumerate(disturbances.items()):
        ax = axes[i]
        levels = dinfo["levels"]
        
        for j, (label, res_dict) in enumerate(results.items()):
            color = colors[j]
            res = res_dict[dtype]
            ax.errorbar(levels, res['m'], yerr=res['s'], fmt='-o', color=color, 
                        label=label, linewidth=2, capsize=3, alpha=0.7)
            
        ax.axhline(0, color='black', linestyle=':', alpha=0.5)
        ax.set_title(dinfo["name"], fontsize=14, fontweight='bold')
        ax.set_xlabel(dinfo["xlabel"], fontsize=12)
        ax.set_ylabel("Discriminator Logit", fontsize=12)
        ax.grid(True, alpha=0.3)
        if dtype == "quant": ax.set_xscale('log'); ax.invert_xaxis()
        ax.text(0.02, 0.95, "REAL-LIKE", transform=ax.transAxes, color='green', fontweight='bold', fontsize=10)
        ax.text(0.02, 0.05, "FAKE-LIKE", transform=ax.transAxes, color='red', fontweight='bold', fontsize=10)

    for ax in axes[len(disturbances):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=10)
    var_display = cfg.selected_variable.replace('_', ' ').title() if len(model_vars) == 1 else "All Fields"
    holdout_names = [dinfo["name"].replace(" (Holdout)", "") for dinfo in disturbances.values() if dinfo.get("held_out")]
    holdout_note = f"\nHoldout corruptions: {', '.join(holdout_names)}" if holdout_names else ""
    plt.suptitle(f"K-Fold Holdout: Sensitivity to Synthetic Artifacts{holdout_note}\nModel: {cfg.model_name} | Variable: {var_display}", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(cfg.output_dir, f"kfold_disturbance_analysis_GT_FINAL_{cfg.model_name}_{variable_tag(cfg)}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')

    # --- Visualization of Corruptions (Grid) ---
    sample_idx = 0
    sample_ds = era5_test_ds.isel(time=sample_idx)
    if cfg.get("level") is not None:
        if "level" in sample_ds.dims:
            sample_ds = sample_ds.sel(level=cfg.get("level"))
        elif "pressure_level" in sample_ds.dims:
            sample_ds = sample_ds.sel(pressure_level=cfg.get("level"))
    v = cfg.selected_variable
    plot_configs = {'u_component_of_wind': 'viridis', 'v_component_of_wind': 'viridis', 'wind_speed': 'viridis', 'temperature': 'inferno', 'specific_humidity': 'GnBu'}
    cmap = plot_configs.get(v, 'viridis')
    channel_idx = model_vars.index(v) if v in model_vars else 0
    sample_channels = []
    for variable in model_vars:
        if variable in sample_ds.data_vars:
            raw = sample_ds[variable].values.astype(np.float32)
            scale = stds[variable] if stds[variable] > 1e-8 else 1.0
            sample_channels.append(np.nan_to_num((raw - means[variable]) / scale, nan=0.0))
        else:
            ref = sample_ds[v] if v in sample_ds.data_vars else list(sample_ds.data_vars.values())[0]
            sample_channels.append(np.zeros(ref.shape, dtype=np.float32))
    clean_sample = torch.tensor(np.stack(sample_channels), dtype=torch.float32)
    n_types, n_levels = len(disturbances), 5
    fig_vis, axes_vis = plt.subplots(n_types, n_levels, figsize=(4 * n_levels, 4 * n_types), subplot_kw={'projection': ccrs.PlateCarree()})
    axes_vis = np.asarray(axes_vis).reshape(n_types, n_levels)
    for i, (dtype, dinfo) in enumerate(disturbances.items()):
        levels = dinfo["levels"][:n_levels]
        for j, level in enumerate(levels):
            ax = axes_vis[i, j]
            corrupted = apply_configured_corruption(clean_sample.clone(), dtype, float(level))
            scale = stds[v] if stds[v] > 1e-8 else 1.0
            data = corrupted[channel_idx].detach().cpu().numpy().astype(np.float32) * scale + means[v]
            da_plot = sample_ds[v].copy(deep=True, data=data)
            da_plot.plot(ax=ax, x="longitude", y="latitude", transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False, robust=True)
            ax.coastlines(); ax.add_feature(cfeature.BORDERS, alpha=0.5)
            if i == 0: ax.set_title(f"Level: {level}", fontsize=14)
            if j == 0: ax.text(-0.1, 0.5, dinfo["name"], transform=ax.transAxes, rotation=90, va='center', ha='right', fontsize=16, fontweight='bold')
    plt.suptitle(f"Disturbance Visuals: {v.replace('_', ' ').title()}", fontsize=22, y=1.02)
    plt.tight_layout()
    vis_output_path = os.path.join(cfg.output_dir, f"kfold_disturbance_visuals_GT_{cfg.model_name}_{variable_tag(cfg)}.png")
    plt.savefig(vis_output_path, dpi=200, bbox_inches='tight')
    print(f"Disturbance grid saved to: {vis_output_path}")

    real_ds_global.close()

if __name__ == "__main__": main()
