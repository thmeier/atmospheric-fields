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

from train_discriminator import WeatherDiscriminator

class SimpleInferenceDataset(Dataset):
    def __init__(self, ds, variables, means, stds, level=None, max_samples=0):
        self.ds = ds
        if level is not None:
            if 'level' in self.ds.dims:
                self.ds = self.ds.sel(level=level)
            elif 'pressure_level' in self.ds.dims:
                self.ds = self.ds.sel(pressure_level=level)
        
        self.variables = variables
        self.means = means
        self.stds = stds
        
        all_times = self.ds.time.values
        if max_samples > 0 and len(all_times) > max_samples:
            # Subsample times uniformly
            indices = np.linspace(0, len(all_times) - 1, max_samples, dtype=int)
            self.times = all_times[indices]
            self.ds = self.ds.isel(time=indices)
        else:
            self.times = all_times

        if "prediction_timedelta" in self.ds.dims:
            lt = self.ds.prediction_timedelta.values
            self.lead_hours = lt.astype('timedelta64[h]').astype(int) if np.issubdtype(lt.dtype, np.timedelta64) else lt
            self.num_leads = len(self.lead_hours)
            self.has_lead_dim = True
        else:
            self.lead_hours = np.array([0])
            self.num_leads = 1
            self.has_lead_dim = False

    def __len__(self):
        return len(self.times) * self.num_leads

    def __getitem__(self, idx):
        t_idx = idx // self.num_leads
        l_idx = idx % self.num_leads
        
        ds_slice = self.ds.isel(time=t_idx)
        if self.has_lead_dim:
            ds_slice = ds_slice.isel(prediction_timedelta=l_idx)
            
        channels = []
        for v in self.variables:
            if v in ds_slice.data_vars:
                val = ds_slice[v].values.astype(np.float32)
                norm = (val - self.means[v]) / (self.stds[v] if self.stds[v] > 1e-8 else 1.0)
                channels.append(np.nan_to_num(norm, nan=0.0))
            else:
                # Zero-fill missing variables
                ref_shape = ds_slice.temperature.shape if 'temperature' in ds_slice.data_vars else list(ds_slice.data_vars.values())[0].shape
                channels.append(np.zeros(ref_shape, dtype=np.float32))

        return torch.tensor(np.stack(channels), dtype=torch.float32), l_idx

def run_inference(dataset, model, cfg, device, desc="Inference"):
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    lead_logits = [[] for _ in range(dataset.num_leads)]
    with torch.no_grad():
        for inputs, l_indices in tqdm(dataloader, desc=desc, leave=False):
            outputs = model(inputs.to(device)).cpu().numpy().flatten()
            for logit, l_idx in zip(outputs, l_indices):
                lead_logits[l_idx.item()].append(logit)
    return dataset.lead_hours, [np.mean(l) for l in lead_logits], [np.std(l) for l in lead_logits]

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_ds_global = xr.open_dataset(cfg.test_real_nc_file)
    model_vars = [cfg.selected_variable]
    
    # Slice ERA5 to test ranges
    combined_real = []
    for r in cfg.test_real_ranges:
        combined_real.append(real_ds_global.sel(time=slice(r[0], r[1])))
    curr_real = xr.concat(combined_real, dim='time')
    
    # Use training range for means/stds
    stats_ds = real_ds_global.sel(time=slice(cfg.train_real_range[0], cfg.train_real_range[1]))
    means = {}
    stds = {}
    for v in model_vars:
        if v in stats_ds.data_vars:
            means[v] = float(stats_ds[v].mean())
            stds[v] = float(stats_ds[v].std())
        else:
            means[v] = 0.0
            stds[v] = 1.0
    
    model = WeatherDiscriminator(len(model_vars), cfg.model_name).to(device)
    model.model.load_state_dict(torch.load(os.path.join(cfg.output_dir, f"weather_discriminator_{cfg.model_name}_{cfg.selected_variable}_lightning.pth"), map_location=device))
    model.eval()

    plt.figure(figsize=(10, 6))
    
    # Real Baseline
    _, r_means, r_stds = run_inference(SimpleInferenceDataset(curr_real, model_vars, means, stds, level=cfg.get("level")), model, cfg, device, "Baseline (ERA5)")
    plt.errorbar([0], [r_means[0]], yerr=[r_stds[0]], fmt='o', label="ERA5 (Ground Truth)", markersize=10, color='black')

    files = cfg.comparison_files
    for label, path in files.items():
        if not os.path.exists(path): 
            print(f"Skipping {label}: file not found at {path}")
            continue
        fake_ds = xr.open_dataset(path)
        
        # Check if selected variable is in dataset
        if cfg.selected_variable not in fake_ds.data_vars:
            print(f"Skipping {label}: Variable {cfg.selected_variable} not found.")
            fake_ds.close()
            continue
            
        # Slice fake to test range
        curr_fake = fake_ds.sel(time=slice(cfg.test_fake_range[0], cfg.test_fake_range[1]))
        if len(curr_fake.time) == 0:
            print(f"Skipping {label}: No samples in test range.")
            continue

        # Fake Inference
        l_hrs, m_l, s_l = run_inference(SimpleInferenceDataset(curr_fake, model_vars, means, stds, level=cfg.get("level")), model, cfg, device, label)
        
        idx = np.argsort(l_hrs)
        plt.errorbar(l_hrs[idx], [m_l[i] for i in idx], yerr=[s_l[i] for i in idx], fmt='-o', label=label)
        fake_ds.close()

    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Lead Time (hours)', fontsize=12)
    plt.ylabel('Discriminator Logit Output', fontsize=12)
    
    # Add indicators for Real vs Fake regions
    plt.text(0.02, 0.95, "REAL-LIKE (Logits > 0)", color='green', fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.02, 0.05, "FAKE-LIKE (Logits < 0)", color='red', fontweight='bold', transform=plt.gca().transAxes)

    var_display = cfg.selected_variable.replace('_', ' ').title()
    plt.title(f"Discriminator Logits vs Lead Time\nModel: {cfg.model_name} | Variable: {var_display}", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    output_path = os.path.join(cfg.output_dir, f"comparison_logits_vs_lead_time_{cfg.model_name}_{cfg.selected_variable}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    real_ds_global.close()
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__": main()
