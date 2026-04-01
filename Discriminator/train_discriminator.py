import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import xarray as xr
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. Dataset Definition (From our previous step)
# ==========================================
class WeatherDiscriminatorDataset(Dataset):
    def __init__(self, nc_path, variables):
        self.variables = variables
        self.ds = xr.open_dataset(nc_path)
        self.times = self.ds.time.values
        
        if "prediction_timedelta" not in self.ds.dims:
            raise ValueError("Dataset is missing the 'prediction_timedelta' dimension.")
            
        # Placeholders for normalization (Calculate these on your dataset!)
        self.means = {var: 0.0 for var in variables} 
        self.stds = {var: 1.0 for var in variables}

    def __len__(self):
        return len(self.times) * 2

    def __getitem__(self, idx):
        num_times = len(self.times)
        
        if idx < num_times:
            # REAL (Lead time 0) -> Label 1.0
            time_idx = idx
            lead_idx = 0  
            label = torch.tensor([1.0], dtype=torch.float32)
        else:
            # FAKE (Lead time 12) -> Label 0.0
            time_idx = idx - num_times
            lead_idx = 1  
            label = torch.tensor([0.0], dtype=torch.float32)

        ds_slice = self.ds.isel(time=time_idx, prediction_timedelta=lead_idx)

        channels = []
        for var in self.variables:
            data_array = ds_slice[var].values
            # Normalization
            norm_data = (data_array - self.means[var]) / self.stds[var]
            
            # Handle NaNs (sometimes present in weather data masks)
            norm_data = np.nan_to_num(norm_data, nan=0.0)
            channels.append(norm_data)
            
        return torch.tensor(np.stack(channels), dtype=torch.float32), label

# ==========================================
# 2. Model Definition
# ==========================================
def get_weather_discriminator(num_weather_channels):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    old_conv = model.conv1
    
    # Replace first layer
    new_conv = nn.Conv2d(
        in_channels=num_weather_channels, 
        out_channels=old_conv.out_channels, 
        kernel_size=old_conv.kernel_size, 
        stride=old_conv.stride, 
        padding=old_conv.padding, 
        bias=old_conv.bias is not None
    )
    
    # Tile RGB weights across new channels
    with torch.no_grad():
        repeat_factor = (num_weather_channels // 3) + 1
        new_weights = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :num_weather_channels, :, :]
        new_conv.weight = nn.Parameter(new_weights * (3.0 / num_weather_channels))
        
    model.conv1 = new_conv
    
    # Replace final head for binary classification (1 logit)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# ==========================================
# 3. Main Training Loop
# ==========================================
def main():
    # --- Configuration ---
    nc_file = "/cluster/courses/pmlr/teams/team07/data/graphcast_1.5deg_2018-01-01_2018-12-31.nc" # Path to your downloaded NetCDF
    variables = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
    batch_size = 16
    epochs = 10
    learning_rate = 1e-4
    
    # Set device (GPU if available, else Apple Silicon MPS, else CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # --- Setup Data ---
    dataset = WeatherDiscriminatorDataset(nc_file, variables)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # --- Setup Model ---
    model = get_weather_discriminator(num_weather_channels=len(variables))
    model = model.to(device)
    
    # --- Setup Loss & Optimizer ---
    criterion = nn.BCEWithLogitsLoss()
    
    # We use a slightly higher learning rate for the new layers we just created, 
    # and a lower one for the pre-trained backbone.
    optimizer = optim.AdamW([
        {'params': model.conv1.parameters(), 'lr': learning_rate * 10},
        {'params': model.fc.parameters(), 'lr': learning_rate * 10},
        {'params': [p for n, p in model.named_parameters() if n not in ['conv1.weight', 'fc.weight', 'fc.bias']], 'lr': learning_rate}
    ])

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # tqdm for a nice progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs) # outputs are raw logits
            
            # Calculate Loss
            loss = criterion(outputs, labels)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy (sigmoid > 0.5 means logit > 0)
            predictions = (outputs > 0.0).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Update progress bar
            current_acc = correct_predictions / total_predictions
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{current_acc:.4f}"})
            
        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct_predictions / len(dataset)
        print(f"\nEpoch {epoch+1} Summary -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}\n")

    # --- Save the Model ---
    save_path = "weather_discriminator_resnet18.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model weights saved to {save_path}")

if __name__ == "__main__":
    main()
