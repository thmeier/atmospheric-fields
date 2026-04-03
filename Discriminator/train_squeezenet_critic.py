import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd

# ==========================================
# 1. Dataset: Paired ERA5 (Real) vs GraphCast (Fake)
# ==========================================
class PairedWeatherDataset(Dataset):
    def __init__(self, gc_nc_path, era5_nc_path, variables):
        self.variables = variables
        
        # Open both datasets lazily
        self.ds_gc = xr.open_dataset(gc_nc_path)
        self.ds_era5 = xr.open_dataset(era5_nc_path)
        
        self.times = self.ds_gc.time.values
        self.lead_times = self.ds_gc.prediction_timedelta.values
        
        self.num_times = len(self.times)
        self.num_leads = len(self.lead_times)
        
        # --- STANDARDIZATION ---
        print("Calculating exact normalization stats from the 2018 GraphCast dataset...")
        self.means = {}
        self.stds = {}
        for var in self.variables:
            # .compute() forces xarray to calculate the actual mathematical mean
            self.means[var] = float(self.ds_gc[var].mean().compute())
            self.stds[var] = float(self.ds_gc[var].std().compute())
            print(f"  {var} -> Mean: {self.means[var]:.2f}, Std: {self.stds[var]:.2f}")
        print("Normalization stats ready!\n")

    def __len__(self):
        # 1 Real sample + 1 Fake sample for every valid forecast step
        return self.num_times * self.num_leads * 2

    def __getitem__(self, idx):
        total_fakes = self.num_times * self.num_leads
        
        # First half of dataset is REAL (1.0), second half is FAKE (0.0)
        is_real = (idx < total_fakes)
        
        if is_real:
            base_idx = idx
            label = torch.tensor([1.0], dtype=torch.float32)
        else:
            base_idx = idx - total_fakes
            label = torch.tensor([0.0], dtype=torch.float32)
            
        time_idx = base_idx // self.num_leads
        lead_idx = base_idx % self.num_leads
        
        # Calculate the EXACT Valid Time to match ground truth
        init_time = self.times[time_idx]
        lead_time = self.lead_times[lead_idx]
        
        # Convert numpy datetime to pandas to safely add the timedelta
        valid_time = pd.to_datetime(init_time) + pd.to_timedelta(lead_time)
        
        if is_real:
            # Pull REAL physics from ERA5 at the exact target Valid Time
            # method='nearest' protects against tiny nanosecond float errors
            ds_slice = self.ds_era5.sel(time=valid_time, method='nearest')
        else:
            # Pull FAKE physics from GraphCast generated from Init Time + Lead Time
            ds_slice = self.ds_gc.isel(time=time_idx, prediction_timedelta=lead_idx)

        channels = []
        for var in self.variables:
            data_array = ds_slice[var].values
            
            # Apply True Z-Score Normalization
            norm_data = (data_array - self.means[var]) / self.stds[var]
            norm_data = np.nan_to_num(norm_data, nan=0.0) # Safety catch for masks
            channels.append(norm_data)
            
        return torch.tensor(np.stack(channels), dtype=torch.float32), label

# ==========================================
# 2. Model: Custom SqueezeNet
# ==========================================
def get_squeezenet_weather(num_weather_channels):
    model = models.squeezenet1_1(weights=None)
    
    # 1. Modify the first convolutional layer
    old_conv = model.features[0]
    new_conv = nn.Conv2d(
        in_channels=num_weather_channels, 
        out_channels=old_conv.out_channels, 
        kernel_size=old_conv.kernel_size, 
        stride=old_conv.stride, 
        padding=old_conv.padding
    )
    
    with torch.no_grad():
        repeat_factor = (num_weather_channels // 3) + 1
        new_weights = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :num_weather_channels, :, :]
        new_conv.weight = nn.Parameter(new_weights * (3.0 / num_weather_channels))
        
    model.features[0] = new_conv
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(512, 1, kernel_size=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    
    return model

# ==========================================
# 3. Main Training Loop
# ==========================================
def main():
    # --- File Paths (TRAINING ON 2018) ---
    gc_file = "/cluster/courses/pmlr/teams/team07/data/graphcast_4steps_4vars_1.5deg_2018.nc" 
    era5_file = "/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc"
    
    # --- 4 Variables ---
    variables = [
        '2m_temperature', 
        '10m_u_component_of_wind', 
        '10m_v_component_of_wind',
        'mean_sea_level_pressure'
    ]
    
    batch_size = 16
    epochs = 10
    learning_rate = 1e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Setup Data ---
    dataset = PairedWeatherDataset(gc_file, era5_file, variables)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # --- Setup Model ---
    model = get_squeezenet_weather(num_weather_channels=len(variables))
    model = model.to(device)
    
    # --- Loss & Optimizer ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # Track accuracy and loss
            running_loss += loss.item() * inputs.size(0)
            predictions = (outputs > 0.0).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{(correct_predictions/total_predictions):.4f}"})
            
        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct_predictions / len(dataset)
        print(f"\nEpoch {epoch+1} Summary -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}\n")

    # --- Save Model ---
    save_path = "weather_discriminator_squeezenet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model saved to {save_path}")

if __name__ == "__main__":
    main()