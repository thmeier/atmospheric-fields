import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Model: Custom SqueezeNet (Must match training exactly)
# ==========================================
def get_squeezenet_weather(num_weather_channels):
    model = models.squeezenet1_1(weights=None)
    old_conv = model.features[0]
    
    new_conv = nn.Conv2d(
        in_channels=num_weather_channels, 
        out_channels=old_conv.out_channels, 
        kernel_size=old_conv.kernel_size, 
        stride=old_conv.stride, 
        padding=old_conv.padding
    )
    model.features[0] = new_conv
    
    old_classifier = model.classifier[1]
    model.classifier[1] = nn.Conv2d(old_classifier.in_channels, 1, kernel_size=1)
    return model

# ==========================================
# 2. Dataset: Evaluator for a Specific Lead Time
# ==========================================
class WeatherEvalDataset(Dataset):
    def __init__(self, gc_test_path, era5_test_path, variables, train_means, train_stds, target_lead_idx):
        self.variables = variables
        self.means = train_means
        self.stds = train_stds
        
        self.ds_gc = xr.open_dataset(gc_test_path)
        self.ds_era5 = xr.open_dataset(era5_test_path)
        
        self.times = self.ds_gc.time.values
        self.target_lead_time = self.ds_gc.prediction_timedelta.values[target_lead_idx]
        self.target_lead_idx = target_lead_idx
        
        self.num_times = len(self.times)

    def __len__(self):
        # We only evaluate the GraphCast (Fake) data for this specific lead time
        return self.num_times

    def __getitem__(self, idx):
        init_time = self.times[idx]
        
        # Pull GraphCast prediction for this specific initial time and lead time
        ds_slice = self.ds_gc.isel(time=idx, prediction_timedelta=self.target_lead_idx)

        channels = []
        for var in self.variables:
            data_array = ds_slice[var].values
            norm_data = (data_array - self.means[var]) / self.stds[var]
            norm_data = np.nan_to_num(norm_data, nan=0.0)
            channels.append(norm_data)
            
        return torch.tensor(np.stack(channels), dtype=torch.float32)

# ==========================================
# 3. Main Evaluation Loop
# ==========================================
def main():
    # --- Configuration ---
    train_file = "/cluster/courses/pmlr/teams/team07/data/graphcast_4steps_4vars_1.5deg_2018.nc"
    test_gc_file = "/cluster/courses/pmlr/teams/team07/data/graphcast_4steps_1.5deg_2019-01.nc"
    test_era5_file = "/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc"
    model_weights_path = "weather_discriminator_squeezenet.pth"
    
    variables = [
        '2m_temperature', 
        '10m_u_component_of_wind', 
        '10m_v_component_of_wind',
        'mean_sea_level_pressure'
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # --- Step 1: Get Training Normalization Stats ---
    print("\nExtracting normalization stats from 2018 Training Data...")
    ds_train = xr.open_dataset(train_file)
    train_means = {}
    train_stds = {}
    for var in variables:
        train_means[var] = float(ds_train[var].mean().compute())
        train_stds[var] = float(ds_train[var].std().compute())
    ds_train.close()
    
    # --- Step 2: Setup Model ---
    model = get_squeezenet_weather(num_weather_channels=len(variables))
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # --- Step 3: Evaluate Each Lead Time ---
    lead_time_labels = ['6h', '12h', '72h', '240h']
    realism_scores = []
    
    print("\nStarting Evaluation on Unseen Jan 2019 Data...")
    
    with torch.no_grad():
        for lead_idx, label in enumerate(lead_time_labels):
            dataset = WeatherEvalDataset(test_gc_file, test_era5_file, variables, train_means, train_stds, lead_idx)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
            
            total_score = 0.0
            total_samples = 0
            
            for inputs in dataloader:
                inputs = inputs.to(device)
                
                # Get raw logits and pass through Sigmoid to get probability (0 to 1)
                logits = model(inputs)
                probs = torch.sigmoid(logits)
                
                # Sum up the probabilities
                total_score += probs.sum().item()
                total_samples += inputs.size(0)
                
            average_realism = total_score / total_samples
            realism_scores.append(average_realism)
            print(f"Lead Time {label} -> Average Realism Score: {average_realism:.4f} (1.0 = Perfectly Real, 0.0 = Obviously Fake)")

    # --- Step 4: Plot the Results ---
    plt.figure(figsize=(8, 5))
    plt.plot(lead_time_labels, realism_scores, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title('GraphCast Realism Score Degradation Over Time (Jan 2019)')
    plt.xlabel('Forecast Lead Time')
    plt.ylabel('Critic Realism Score (Probability of being Real)')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = "realism_degradation_plot.png"
    plt.savefig(plot_path)
    print(f"\nEvaluation Complete! Plot saved to {plot_path}")

if __name__ == "__main__":
    main()