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

class DisturbanceInferenceDataset(Dataset):
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
    model.eval()
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs.to(device)).cpu().numpy().flatten()
            logits.extend(outputs)
    return np.mean(logits), np.std(logits)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_vars = [cfg.selected_variable]
    NUMERICAL_MODELS = ["IFS HRES", "ERA5 Forecast"]
    
    # Load ERA5 for disturbance application
    real_ds_global = safe_open_dataset(cfg.test_real_nc_file)
    stats_ds = real_ds_global.sel(time=slice(cfg.train_real_range[0], cfg.train_real_range[1]))
    means = {v: float(stats_ds[v].mean()) for v in model_vars}
    stds = {v: float(stats_ds[v].std()) for v in model_vars}
    
    combined_real = []
    for r in cfg.test_real_ranges:
        combined_real.append(real_ds_global.sel(time=slice(r[0], r[1])))
    era5_test_ds = xr.concat(combined_real, dim='time')

    disturbances = {
        "blur": {"name": "Gaussian Blur", "levels": [0, 0.5, 1.5, 3.0, 5.0], "xlabel": "Sigma"},
        "noise": {"name": "HF Noise", "levels": [0, 0.1, 0.3, 0.6, 1.0], "xlabel": "Amplitude (rel to std)"},
        "grf": {"name": "GRF Noise", "levels": [0, 0.1, 0.3, 0.6, 1.0], "xlabel": "Amplitude (rel to std)"},
        "replace": {"name": "Pixel Replace", "levels": [0, 0.05, 0.15, 0.3, 0.5], "xlabel": "Fraction"},
        "patch": {"name": "Patch Dropout (OOD)", "levels": [0, 0.05, 0.15, 0.3, 0.5], "xlabel": "Fraction"},
        "quant": {"name": "Quantization (OOD)", "levels": [256, 64, 32, 16, 8, 4], "xlabel": "Levels"}
    }

    model = WeatherDiscriminator(len(model_vars), cfg.model_name).to(device)
    
    # Collect all available models: Holdouts and Full Pool
    # Filter to only AI models for holdouts (Numerical models like HRES should not have holdouts)
    model_files = [f for f in os.listdir(cfg.output_dir) if f.endswith(".pth") and (f"_exclude_" in f or "_full_pool" in f)]
    
    results = {} # label -> dtype -> {'m': [], 's': []}

    for m_file in model_files:
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
        model.model.load_state_dict(torch.load(os.path.join(cfg.output_dir, m_file), map_location=device))
        results[label] = {}
        
        for dtype, dinfo in disturbances.items():
            results[label][dtype] = {'m': [], 's': []}
            for level in dinfo["levels"]:
                ds_disturbed = DisturbanceInferenceDataset(era5_test_ds, model_vars, means, stds, dtype, level, level=cfg.get("level"), max_samples=50)
                m, s = get_mean_logit(ds_disturbed, model, cfg, device)
                results[label][dtype]['m'].append(m)
                results[label][dtype]['s'].append(s)

    # --- Plotting Line Graphs ---
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    axes = axes.flatten()
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

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=10)
    var_display = cfg.selected_variable.replace('_', ' ').title()
    plt.suptitle(f"K-Fold Holdout: Sensitivity to Synthetic Artifacts\nNote: Patch & Quantization are Out-Of-Distribution (not seen during training)\nModel: {cfg.model_name} | Variable: {var_display}", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(cfg.output_dir, f"kfold_disturbance_analysis_GT_FINAL_{cfg.model_name}_{cfg.selected_variable}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')

    # --- Visualization of Corruptions (Grid) ---
    sample_idx = 0
    sample_ds = era5_test_ds.isel(time=sample_idx)
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
    vis_output_path = os.path.join(cfg.output_dir, f"kfold_disturbance_visuals_GT_{cfg.model_name}_{cfg.selected_variable}.png")
    plt.savefig(vis_output_path, dpi=200, bbox_inches='tight')
    print(f"Disturbance grid saved to: {vis_output_path}")

    real_ds_global.close()

if __name__ == "__main__": main()
