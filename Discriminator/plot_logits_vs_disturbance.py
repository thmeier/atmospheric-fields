import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import os
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from train_discriminator import WeatherDiscriminator, safe_open_dataset

# ==========================================
# Disturbance Functions
# ==========================================

def apply_gaussian_blur(data, sigma):
    if sigma == 0: return data
    return gaussian_filter(data, sigma=sigma)

def apply_hf_noise(data, std_dev):
    if std_dev == 0: return data
    noise = np.random.normal(0, std_dev, data.shape)
    return data + noise

def apply_grf_noise(data, amplitude, length_scale=2.0):
    if amplitude == 0: return data
    white_noise = np.random.normal(0, 1, data.shape)
    grf = gaussian_filter(white_noise, sigma=length_scale)
    grf = (grf - grf.mean()) / (grf.std() + 1e-8)
    return data + amplitude * grf

def apply_pixel_replace(data, fraction):
    if fraction == 0: return data
    flat_data = data.flatten()
    n_replace = int(fraction * len(flat_data))
    indices = np.random.choice(len(flat_data), n_replace, replace=False)
    flat_data[indices] = np.random.normal(np.mean(data), np.std(data), n_replace)
    return flat_data.reshape(data.shape)

def apply_patch_dropout(data, fraction, patch_size=8):
    if fraction == 0: return data
    h, w = data.shape
    data_out = data.copy()
    n_h, n_w = h // patch_size, w // patch_size
    n_total = n_h * n_w
    n_drop = int(fraction * n_total)
    indices = np.random.choice(n_total, n_drop, replace=False)
    for idx in indices:
        r, c = (idx // n_w) * patch_size, (idx % n_w) * patch_size
        data_out[r:r+patch_size, c:c+patch_size] = np.mean(data)
    return data_out

def apply_quantization(data, n_levels):
    if n_levels >= 256 or n_levels <= 0: return data
    d_min, d_max = data.min(), data.max()
    if d_max == d_min: return data
    data_norm = (data - d_min) / (d_max - d_min)
    data_q = np.round(data_norm * (n_levels - 1)) / (n_levels - 1)
    return data_q * (d_max - d_min) + d_min

# ==========================================
# Dataset for Disturbance Analysis
# ==========================================

class DisturbanceDataset(Dataset):
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
                
                if self.disturbance_type == "blur":
                    disturbed = apply_gaussian_blur(raw_data, self.severity)
                elif self.disturbance_type == "noise":
                    disturbed = apply_hf_noise(raw_data, self.severity * self.stds[v])
                elif self.disturbance_type == "grf":
                    disturbed = apply_grf_noise(raw_data, self.severity * self.stds[v])
                elif self.disturbance_type == "replace":
                    disturbed = apply_pixel_replace(raw_data, self.severity)
                elif self.disturbance_type == "patch":
                    disturbed = apply_patch_dropout(raw_data, self.severity)
                elif self.disturbance_type == "quant":
                    disturbed = apply_quantization(raw_data, self.severity)
                else:
                    disturbed = raw_data
                
                norm = (disturbed - self.means[v]) / (self.stds[v] if self.stds[v] > 1e-8 else 1.0)
                channels.append(np.nan_to_num(norm, nan=0.0))
            else:
                ref_shape = ds_slice.temperature.shape if 'temperature' in ds_slice.data_vars else list(ds_slice.data_vars.values())[0].shape
                channels.append(np.zeros(ref_shape, dtype=np.float32))

        return torch.tensor(np.stack(channels), dtype=torch.float32)

def get_mean_logit(dataset, model, cfg, device):
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    logits = []
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs.to(device)).cpu().numpy().flatten()
            logits.extend(outputs)
    return np.mean(logits), np.std(logits)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running disturbance analysis on {device}...")

    real_ds_global = safe_open_dataset(cfg.test_real_nc_file)
    combined_real = []
    for r in cfg.test_real_ranges:
        combined_real.append(real_ds_global.sel(time=slice(r[0], r[1])))
    test_real_ds = xr.concat(combined_real, dim='time')

    model_vars = [cfg.selected_variable]
    stats_ds = real_ds_global.sel(time=slice(cfg.train_real_range[0], cfg.train_real_range[1]))
    means = {v: float(stats_ds[v].mean()) for v in model_vars}
    stds = {v: float(stats_ds[v].std()) for v in model_vars}

    model = WeatherDiscriminator(len(model_vars), cfg.model_name).to(device)
    model_path = os.path.join(cfg.output_dir, f"weather_discriminator_{cfg.model_name}_{cfg.selected_variable}_lightning.pth")
    if os.path.exists(model_path):
        model.model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    disturbances = {
        "blur": {"name": "Gaussian Blur", "levels": [0, 0.5, 1.5, 3.0, 5.0], "xlabel": "Sigma"},
        "noise": {"name": "HF Noise", "levels": [0, 0.1, 0.3, 0.6, 1.0], "xlabel": "Amplitude (rel to std)"},
        "grf": {"name": "GRF Noise", "levels": [0, 0.1, 0.3, 0.6, 1.0], "xlabel": "Amplitude (rel to std)"},
        "replace": {"name": "Pixel Replace", "levels": [0, 0.05, 0.15, 0.3, 0.5], "xlabel": "Fraction"},
        "patch": {"name": "Patch Dropout (OOD)", "levels": [0, 0.05, 0.15, 0.3, 0.5], "xlabel": "Fraction"},
        "quant": {"name": "Quantization (OOD)", "levels": [256, 64, 32, 16, 8, 4], "xlabel": "Levels"}
    }

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    axes = axes.flatten()

    for i, (dtype, dinfo) in enumerate(disturbances.items()):
        ax = axes[i]
        m_logits, s_logits = [], []
        
        for level in tqdm(dinfo["levels"], desc=dinfo["name"]):
            dataset = DisturbanceDataset(test_real_ds, model_vars, means, stds, dtype, level, level=cfg.get("level"), max_samples=cfg.max_samples if cfg.max_samples > 0 else 100)
            m, s = get_mean_logit(dataset, model, cfg, device)
            m_logits.append(m)
            s_logits.append(s)
        
        ax.errorbar(dinfo["levels"], m_logits, yerr=s_logits, fmt='-o', capsize=5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_title(dinfo["name"], fontsize=14, fontweight='bold')
        ax.set_xlabel(dinfo["xlabel"], fontsize=12)
        ax.set_ylabel("Discriminator Logit", fontsize=12)
        ax.grid(True, alpha=0.3)
        if dtype == "quant": 
            ax.set_xscale('log')
            ax.invert_xaxis()

    fig.tight_layout()
    analysis_output_path = os.path.join(cfg.output_dir, f"disturbance_analysis_{cfg.model_name}_{cfg.selected_variable}.png")
    fig.savefig(analysis_output_path, dpi=200, bbox_inches='tight')

    # --- Visualization of Corruptions (Grid) ---
    sample_idx = 0
    sample_ds = test_real_ds.isel(time=sample_idx)
    v = cfg.selected_variable
    plot_configs = {'u_component_of_wind': 'viridis', 'v_component_of_wind': 'viridis', 'wind_speed': 'viridis', 'temperature': 'inferno', 'specific_humidity': 'GnBu'}
    cmap = plot_configs.get(v, 'viridis')
    raw_sample = sample_ds[v].values.astype(np.float32)
    n_types, n_levels = len(disturbances), 5
    fig_vis, axes_vis = plt.subplots(n_types, n_levels, figsize=(4 * n_levels, 4 * n_types), subplot_kw={'projection': ccrs.PlateCarree()})
    for i, (dtype, dinfo) in enumerate(disturbances.items()):
        levels = dinfo["levels"][:n_levels]
        for j, level in enumerate(levels):
            ax = axes_vis[i, j]
            if dtype == "blur": data = apply_gaussian_blur(raw_sample, level)
            elif dtype == "noise": data = apply_hf_noise(raw_sample, level * stds[v])
            elif dtype == "grf": data = apply_grf_noise(raw_sample, level * stds[v])
            elif dtype == "replace": data = apply_pixel_replace(raw_sample, level)
            elif dtype == "patch": data = apply_patch_dropout(raw_sample, level)
            elif dtype == "quant": data = apply_quantization(raw_sample, level)
            else: data = raw_sample
            da_plot = sample_ds[v].copy(deep=True, data=data)
            da_plot.plot(ax=ax, x="longitude", y="latitude", transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False, robust=True)
            ax.coastlines(); ax.add_feature(cfeature.BORDERS, alpha=0.5)
            if i == 0: ax.set_title(f"Level: {level}", fontsize=14)
            if j == 0: ax.text(-0.1, 0.5, dinfo["name"], transform=ax.transAxes, rotation=90, va='center', ha='right', fontsize=16, fontweight='bold')
    plt.suptitle(f"Disturbance Visuals: {v.replace('_', ' ').title()}", fontsize=22, y=1.02)
    plt.tight_layout()
    vis_output_path = os.path.join(cfg.output_dir, f"disturbance_visuals_{cfg.model_name}_{cfg.selected_variable}.png")
    plt.savefig(vis_output_path, dpi=200, bbox_inches='tight')
    print(f"Disturbance grid examples saved to: {vis_output_path}")

    real_ds_global.close()

if __name__ == "__main__": main()
