import torch
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import the Dataset and Model classes from your training script
# (Assuming you saved the previous code as train_discriminator.py)
from train_discriminator import WeatherDiscriminatorDataset, get_weather_discriminator

def evaluate_and_visualize():
    # --- Configuration ---
    test_nc_file = "/cluster/courses/pmlr/teams/team07/data/graphcast_1.5deg_2019-01-01_2019-02-01.nc"  # Your separate test file
    model_weights = "~/era5_data_handling/weather_discriminator_resnet18.pth"
    model_vars = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
    #plot_vars = ['wind_speed', '2m_temperature', 'mean_sea_level_pressure']
    plot_vars = ['wind_speed', '2m_temperature']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # --- Setup Data and Model ---
    test_dataset = WeatherDiscriminatorDataset(test_nc_file, model_vars)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = get_weather_discriminator(num_weather_channels=len(model_vars))
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device)
    model.eval()

    # --- 1. Run Inference ---
    results = []
    print("Running inference on test set...")
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            logit = model(inputs).item()
            label = labels.item()
            
            results.append({
                'idx': idx,
                'logit': logit,
                'label': label
            })

    # --- 2. Analyze Logits ---
    uncertain = sorted(results, key=lambda x: abs(x['logit']))[:3]
    fooled = sorted([r for r in results if r['label'] == 0.0], key=lambda x: x['logit'], reverse=True)[:3]
    obvious = sorted([r for r in results if r['label'] == 0.0], key=lambda x: x['logit'])[:3]

    print(f"\nTop 'Fooled' Logit: {fooled[0]['logit']:.2f}")
    print(f"Worst 'Obvious' Fake Logit: {obvious[0]['logit']:.2f}")

    # --- 3. Generate 3x3 Cartopy Grids for Each Variable ---
    print("\nGenerating 3x3 geographic visualizations...")
    ds = xr.open_dataset(test_nc_file)
    num_times = len(test_dataset.times)

    categories = [
        ("Most Uncertain\n(Near 0 Logit)", uncertain),
        ("Best Fakes\n", fooled),
        ("Worst Fakes\n", obvious)
    ]

    # Configuration for how each variable should look
    plot_configs = {
        'wind_speed': {'cmap': 'viridis', 'title': '10m Wind Speed', 'unit': 'm/s'},
        '2m_temperature': {'cmap': 'inferno', 'title': '2m Temperature', 'unit': 'K'},
        #'mean_sea_level_pressure': {'cmap': 'coolwarm', 'title': 'Mean Sea Level Pressure', 'unit': 'Pa'}
    }

    # Loop through each variable to create a separate 3x3 file
    for var_name in plot_vars:
        print(f"Plotting {var_name}...")
        config = plot_configs[var_name]
        
        # Initialize figure with PlateCarree projection for all subplots
        fig, axes = plt.subplots(
            3, 3, 
            figsize=(18, 15), 
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        fig.suptitle(f"Discriminator Evaluation: {config['title']}", fontsize=20, y=0.98)

        for row_idx, (cat_name, cat_list) in enumerate(categories):
            for col_idx, item in enumerate(cat_list):
                ax = axes[row_idx, col_idx]
                global_idx = item['idx']
                logit = item['logit']
                
                # Reverse-engineer the dataset index
                if global_idx < num_times:
                    time_idx = global_idx
                    lead_idx = 0
                    data_type = "REAL (ERA5)"
                else:
                    time_idx = global_idx - num_times
                    lead_idx = 1
                    data_type = "FAKE (Gen)"

                # Extract map slice
                map_slice = ds.isel(time=time_idx, prediction_timedelta=lead_idx)
                
                # Handle derived variables vs standard variables
                if var_name == 'wind_speed':
                    u = map_slice['10m_u_component_of_wind']
                    v = map_slice['10m_v_component_of_wind']
                    plot_data = (u**2 + v**2)**0.5
                else:
                    plot_data = map_slice[var_name]
                
                # Plot using Cartopy transform
                im = plot_data.plot(
                    ax=ax, 
                    x="longitude",
                    y="latitude",
                    transform=ccrs.PlateCarree(),
                    cmap=config['cmap'], 
                    robust=True, 
                    add_colorbar=False
                )
                
                # Add Geographic Features
                ax.coastlines(color='black', linewidth=0.8)
                ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
                
                # Clean up the axis
                init_time = str(test_dataset.times[time_idx]).split('T')[0]
                ax.set_title(f"True: {data_type} | Logit: {logit:.2f}\nInit: {init_time}", fontsize=11)
                
                if col_idx == 0:
                    # Add row labels as text since ylabel conflicts with cartopy gridlines
                    ax.text(-0.15, 0.5, cat_name, va='bottom', ha='center',
                            rotation='vertical', rotation_mode='anchor',
                            transform=ax.transAxes, fontsize=14, fontweight='bold')

            # Add a dedicated colorbar for each row
            cbar = fig.colorbar(im, ax=axes[row_idx, :].ravel().tolist(), location="bottom", pad=0.02, shrink=0.6)
            cbar.set_label(f"{config['title']} ({config['unit']})")

        plt.tight_layout(rect=[0.05, 0.03, 1, 0.95]) 
        filename = f"discriminator_3x3_{var_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig) # Clear memory before the next variable

    print("All visualizations saved successfully.")

if __name__ == "__main__":
    evaluate_and_visualize()
