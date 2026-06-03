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

from train_discriminator import WeatherDiscriminator, WeatherDiscriminatorDataset, safe_open_dataset

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
                ref_shape = ds_slice.temperature.shape if 'temperature' in ds_slice.data_vars else list(ds_slice.data_vars.values())[0].shape
                channels.append(np.zeros(ref_shape, dtype=np.float32))

        return torch.tensor(np.stack(channels), dtype=torch.float32), l_idx

def run_inference(dataset, model, cfg, device, desc="Inference"):
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    lead_logits = [[] for _ in range(dataset.num_leads)]
    model.eval()
    with torch.no_grad():
        for inputs, l_indices in tqdm(dataloader, desc=desc, leave=False):
            outputs = model(inputs.to(device)).cpu().numpy().flatten()
            for logit, l_idx in zip(outputs, l_indices):
                lead_logits[l_idx.item()].append(logit)
    return dataset.lead_hours, [np.mean(l) for l in lead_logits], [np.std(l) for l in lead_logits]

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_ds_global = safe_open_dataset(cfg.test_real_nc_file)
    model_vars = [cfg.selected_variable]
    NUMERICAL_MODELS = ["IFS HRES", "ERA5 Forecast"]
    
    # Slice ERA5 to test ranges
    combined_real = []
    for r in cfg.test_real_ranges:
        combined_real.append(real_ds_global.sel(time=slice(r[0], r[1])))
    curr_real = xr.concat(combined_real, dim='time')
    
    # Use training range for means/stds
    stats_ds = real_ds_global.sel(time=slice(cfg.train_real_range[0], cfg.train_real_range[1]))
    means = {v: float(stats_ds[v].mean()) for v in model_vars}
    stds = {v: float(stats_ds[v].std()) for v in model_vars}
    
    model = WeatherDiscriminator(len(model_vars), cfg.model_name).to(device)
    era5_dataset = SimpleInferenceDataset(curr_real, model_vars, means, stds, level=cfg.get("level"))
    
    # Pre-load all datasets
    ml_datasets = {}
    num_datasets = {}
    max_l_hr = 0
    for label, path in cfg.comparison_files.items():
        if not os.path.exists(path): continue
        ds = safe_open_dataset(path)
        if cfg.selected_variable not in ds.data_vars:
            ds.close(); continue
        curr_ds = ds.sel(time=slice(cfg.test_fake_range[0], cfg.test_fake_range[1]))
        if len(curr_ds.time) == 0:
            ds.close(); continue
        
        inf_ds = SimpleInferenceDataset(curr_ds, model_vars, means, stds, level=cfg.get("level"))
        if inf_ds.lead_hours.max() > max_l_hr:
            max_l_hr = inf_ds.lead_hours.max()
            
        if label in NUMERICAL_MODELS:
            num_datasets[label] = inf_ds
        else:
            ml_datasets[label] = inf_ds

    # --- Analysis Loop ---
    plot_results = []
    
    for ml_label, ml_inf_ds in ml_datasets.items():
        safe_label = ml_label.replace(' ', '_').replace('/', '_')
        kfold_model_path = os.path.join(cfg.output_dir, f"discriminator_{cfg.model_name}_{cfg.selected_variable}_exclude_{safe_label}.pth")
        
        if not os.path.exists(kfold_model_path):
            continue
            
        model.model.load_state_dict(torch.load(kfold_model_path, map_location=device))
        
        # 1. Evaluate the held-out ML model
        l_hrs, m_ml, s_ml = run_inference(ml_inf_ds, model, cfg, device, f"Holdout -> {ml_label}")
        
        # 2. Evaluate all numerical models on THIS discriminator
        all_num_m = []
        all_num_s = []
        for num_label, num_inf_ds in num_datasets.items():
            _, m_n, s_n = run_inference(num_inf_ds, model, cfg, device, f"Num ({num_label}) on {ml_label}-holdout")
            all_num_m.append(m_n)
            all_num_s.append(s_n)
        
        # Mean of means and pooled standard deviation
        avg_num_m = np.mean(all_num_m, axis=0)
        avg_num_s = np.mean(all_num_s, axis=0)
        
        # 3. Evaluate ERA5 on THIS discriminator
        _, m_e, s_e = run_inference(era5_dataset, model, cfg, device, f"ERA5 on {ml_label}-holdout")
        
        plot_results.append({
            'label': ml_label,
            'l_hrs': l_hrs,
            'ml_mean': m_ml,
            'ml_std': s_ml,
            'num_mean': avg_num_m,
            'num_std': avg_num_s,
            'era5_mean': m_e[0],
            'era5_std': s_e[0]
        })

    # --- Plotting ---
    plt.figure(figsize=(15, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_results)))
    
    # Global Ground Truth Baseline
    avg_era5_m = np.mean([r['era5_mean'] for r in plot_results])
    avg_era5_s = np.mean([r['era5_std'] for r in plot_results])
    
    # Draw ERA5 Shaded Band
    plt.axhline(avg_era5_m, color='black', linewidth=1.5, label="ERA5 (Reference Mean)", zorder=10)
    plt.fill_between([-5, max_l_hr + 5], avg_era5_m - avg_era5_s, avg_era5_m + avg_era5_s, 
                     color='black', alpha=0.1, label="ERA5 $\pm$ 1 $\sigma$", zorder=1)
    
    # Ground Truth Point at 0 with very distinct error bars
    plt.errorbar([0], [avg_era5_m], yerr=[avg_era5_s], fmt='o', color='black', 
                 markersize=8, capsize=10, elinewidth=3, capthick=3, label='_nolegend_', zorder=11)

    for i, res in enumerate(plot_results):
        l_hrs = res['l_hrs']
        idx = np.argsort(l_hrs)
        color = colors[i]
        
        m_ml = np.array([res['ml_mean'][j] for j in idx])
        s_ml = np.array([res['ml_std'][j] for j in idx])
        m_num = np.array([res['num_mean'][j] for j in idx])
        s_num = np.array([res['num_std'][j] for j in idx])
        l_hrs_sorted = l_hrs[idx]

        # ML Model (Solid with error bars)
        plt.errorbar(l_hrs_sorted, m_ml, yerr=s_ml, fmt='-o', 
                     color=color, linewidth=2, capsize=4, capthick=1.5, alpha=0.9,
                     label=f"ML: {res['label']} (Holdout)")
        
        # Numerical Average (Dashed with error bars)
        plt.errorbar(l_hrs_sorted, m_num, yerr=s_num, fmt='--s', 
                     color=color, linewidth=1.5, capsize=3, capthick=1, alpha=0.6,
                     label=f"Numerical Avg (on {res['label']} Discr)")

    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.xlim(-5, max_l_hr + 5)
    plt.xlabel('Lead Time (hours)', fontsize=12)
    plt.ylabel('Discriminator Logit Output', fontsize=12)
    plt.text(0.02, 0.96, "REAL-LIKE (Logits > 0)", color='green', fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.02, 0.04, "FAKE-LIKE (Logits < 0)", color='red', fontweight='bold', transform=plt.gca().transAxes)

    var_display = cfg.selected_variable.replace('_', ' ').title()
    plt.title(f"K-Fold Holdout Analysis: Neural vs Numerical Generalization\nModel: {cfg.model_name} | Variable: {var_display}", fontsize=14)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=9, ncol=1)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    output_path = os.path.join(cfg.output_dir, f"kfold_comparison_ML_vs_Numerical_{cfg.model_name}_{cfg.selected_variable}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    real_ds_global.close()

if __name__ == "__main__": main()
