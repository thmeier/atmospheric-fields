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
    def __init__(self, ds, variables, means, stds):
        self.ds = ds
        self.variables = variables
        self.means = means
        self.stds = stds
        self.times = self.ds.time.values
        
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
            
        channels = [np.nan_to_num((ds_slice[v].values.astype(np.float32) - self.means[v]) / (self.stds[v] if self.stds[v] > 1e-8 else 1.0), nan=0.0) for v in self.variables]
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
    means = {v: float(real_ds_global[v].mean()) for v in model_vars}
    stds = {v: float(real_ds_global[v].std()) for v in model_vars}
    
    model = WeatherDiscriminator(1, cfg.model_name).to(device)
    model.model.load_state_dict(torch.load(os.path.join(cfg.output_dir, f"weather_discriminator_{cfg.model_name}_lightning.pth"), map_location=device))
    model.eval()

    plt.figure(figsize=(10, 6))
    files = cfg.get("comparison_files", {"Default": cfg.test_fake_nc_file})
    if not isinstance(files, (dict, DictConfig)):
        files = {"Default": files}

    for label, path in files.items():
        if not os.path.exists(path): continue
        fake_ds = xr.open_dataset(path)
        
        # Calculate common range for this specific file
        t_min = max(real_ds_global.time.min(), fake_ds.time.min())
        t_max = min(real_ds_global.time.max(), fake_ds.time.max())
        
        # Slice both to common range
        curr_real = real_ds_global.sel(time=slice(t_min, t_max))
        curr_fake = fake_ds.sel(time=slice(t_min, t_max))
        
        if len(curr_real.time) == 0 or len(curr_fake.time) == 0:
            print(f"Skipping {label}: No overlapping time range.")
            continue

        # Real Baseline (t=0) for this range
        _, r_means, r_stds = run_inference(SimpleInferenceDataset(curr_real, model_vars, means, stds), model, cfg, device, f"Baseline {label}")
        t0_mean, t0_std = r_means[0], r_stds[0]

        # Fake Inference
        l_hrs, m_l, s_l = run_inference(SimpleInferenceDataset(curr_fake, model_vars, means, stds), model, cfg, device, label)
        
        leads, ms, ss = (np.concatenate([[0], l_hrs]), np.concatenate([[t0_mean], m_l]), np.concatenate([[t0_std], s_l])) if 0 not in l_hrs else (l_hrs, m_l, s_l)
        idx = np.argsort(leads); plt.errorbar(leads[idx], ms[idx], yerr=ss[idx], fmt='-o', label=label)
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
