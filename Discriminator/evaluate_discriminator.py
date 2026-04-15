import torch
import torch.nn as nn
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import hydra
from omegaconf import DictConfig

# Import the Dataset and Model classes from your training script
from train_discriminator import WeatherDiscriminatorDataset, WeatherDiscriminator

import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def evaluate_and_visualize(cfg: DictConfig):
    # --- Configuration ---
    real_nc_file = cfg.test_real_nc_file
    fake_nc_file = cfg.test_fake_nc_file
    
    model_weights = os.path.join(cfg.output_dir, f"weather_discriminator_{cfg.model_name}_lightning.pth")
    model_vars = [cfg.selected_variable]
    plot_vars = [cfg.selected_variable]
    if cfg.selected_variable == '2m_temperature' and '10m_u_component_of_wind' in cfg.variables and '10m_v_component_of_wind' in cfg.variables:
        plot_vars.append('wind_speed')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # --- Setup Data and Model ---
    test_dataset = WeatherDiscriminatorDataset(real_nc_file, fake_nc_file, model_vars)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    lightning_model = WeatherDiscriminator(num_weather_channels=1, model_name=cfg.model_name)
    lightning_model.model.load_state_dict(torch.load(model_weights, map_location=device))
    model = lightning_model.to(device).eval()

    # --- 1. Run Inference ---
    results = []
    criterion = nn.BCEWithLogitsLoss()
    running_loss, correct_predictions = 0.0, 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Inference")
        for idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            correct_predictions += ((outputs > 0.0).float() == labels).sum().item()
            results.append({'idx': idx, 'logit': outputs.item(), 'label': labels.item()})

    print(f"\nOverall Test Accuracy: {correct_predictions / len(test_dataset):.4f}")

    # --- 2. Analyze Logits ---
    realest = sorted(results, key=lambda x: -x['logit'])[:3]
    uncertain = sorted(results, key=lambda x: abs(x['logit']))[:3]
    fooled = sorted([r for r in results if r['label'] == 0.0], key=lambda x: x['logit'], reverse=True)[:3]
    obvious = sorted([r for r in results if r['label'] == 0.0], key=lambda x: x['logit'])[:3]

    categories = [("Realest", realest), ("Most Uncertain", uncertain), ("Best Fakes", fooled), ("Worst Fakes", obvious)]
    plot_configs = {
        '10m_u_component_of_wind': {'cmap': 'viridis', 'title': '10m u wind', 'unit': 'm/s'},
        '10m_v_component_of_wind': {'cmap': 'viridis', 'title': '10m v wind', 'unit': 'm/s'},
        'wind_speed': {'cmap': 'viridis', 'title': '10m Wind Speed', 'unit': 'm/s'},
        '2m_temperature': {'cmap': 'inferno', 'title': '2m Temperature', 'unit': 'K'},
    }

    # --- 3. Visualization ---
    for var_name in plot_vars:
        config = plot_configs[var_name]
        fig, axes = plt.subplots(4, 3, figsize=(18, 15), subplot_kw={'projection': ccrs.PlateCarree()})
        fig.suptitle(f"Discriminator Evaluation: {config['title']}", fontsize=20, y=0.98)

        for row_idx, (cat_name, cat_list) in enumerate(categories):
            for col_idx, item in enumerate(cat_list):
                ax = axes[row_idx, col_idx]
                global_idx, logit = item['idx'], item['logit']
                
                # Logic matched to train_discriminator.py
                samples_per_class = test_dataset.samples_per_class
                is_fake = global_idx >= samples_per_class
                sub_idx = global_idx % samples_per_class
                
                if not is_fake:
                    ds_to_plot = test_dataset.real_ds
                    time_val = test_dataset.real_times[sub_idx % test_dataset.total_real_samples]
                    lead_idx = test_dataset.real_lead_idx
                    data_type = "REAL"
                else:
                    ds_to_plot = test_dataset.fake_ds
                    f_idx = sub_idx % test_dataset.total_fake_samples
                    time_idx = f_idx // test_dataset.num_fake_leads
                    time_val = test_dataset.fake_times[time_idx]
                    lead_inner_idx = f_idx % test_dataset.num_fake_leads
                    lead_idx = test_dataset.fake_lead_indices[lead_inner_idx]
                    data_type = "FAKE"

                map_slice = ds_to_plot.sel(time=time_val)
                if lead_idx is not None:
                    map_slice = map_slice.isel(prediction_timedelta=lead_idx)
                
                plot_data = (map_slice['10m_u_component_of_wind']**2 + map_slice['10m_v_component_of_wind']**2)**0.5 if var_name == 'wind_speed' else map_slice[var_name]
                plot_data.plot(ax=ax, x="longitude", y="latitude", transform=ccrs.PlateCarree(), cmap=config['cmap'], robust=True, add_colorbar=False)
                ax.coastlines(); ax.add_feature(cfeature.BORDERS, alpha=0.5)
                ax.set_title(f"{data_type} | Logit: {logit:.2f}\n{str(time_val)[:13]}", fontsize=10)

        plt.savefig(os.path.join(cfg.output_dir, f"evaluation_{cfg.model_name}_{var_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    evaluate_and_visualize()
