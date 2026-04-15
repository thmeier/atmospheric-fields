import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import xarray as xr
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

# Load environment variables from .env file
load_dotenv("../wandb_info.env")

# ==========================================
# 1. Dataset Definition
# ==========================================
class WeatherDiscriminatorDataset(Dataset):
    def __init__(self, real_nc_path, fake_nc_path, variables, means=None, stds=None):
        self.variables = variables
        self.real_ds = xr.open_dataset(real_nc_path)
        self.fake_ds = xr.open_dataset(fake_nc_path)
        
        # Restrict to common time range [t_min, t_max]
        t_min = max(self.real_ds.time.min(), self.fake_ds.time.min())
        t_max = min(self.real_ds.time.max(), self.fake_ds.time.max())
        
        self.real_ds = self.real_ds.sel(time=slice(t_min, t_max))
        self.fake_ds = self.fake_ds.sel(time=slice(t_min, t_max))
        
        self.real_times = self.real_ds.time.values
        self.fake_times = self.fake_ds.time.values
        
        print(f"Dataset Initialized (Time Range: {str(t_min.values)[:13]} to {str(t_max.values)[:13]}):")
        print(f"  REAL samples: {len(self.real_times)} time steps.")
        print(f"  FAKE samples: {len(self.fake_times)} time steps.")

        # Identify lead indices for REAL (0h in real_ds)
        if "prediction_timedelta" in self.real_ds.dims:
            real_leads = self.real_ds.prediction_timedelta.values
            real_lead_hours = real_leads.astype('timedelta64[h]').astype(int) if np.issubdtype(real_leads.dtype, np.timedelta64) else real_leads
            zero_indices = np.where(real_lead_hours == 0)[0]
            self.real_lead_idx = int(zero_indices[0]) if len(zero_indices) > 0 else 0
        else:
            self.real_lead_idx = None

        # Identify ALL lead indices for FAKE > 0h
        if "prediction_timedelta" in self.fake_ds.dims:
            fake_leads = self.fake_ds.prediction_timedelta.values
            fake_lead_hours = fake_leads.astype('timedelta64[h]').astype(int) if np.issubdtype(fake_leads.dtype, np.timedelta64) else fake_leads
            self.fake_lead_indices = np.where(fake_lead_hours > 0)[0].tolist()
            if len(self.fake_lead_indices) == 0:
                raise ValueError(f"No lead times > 0h found in {fake_nc_path}")
            print(f"Using {len(self.fake_lead_indices)} FAKE lead times: {fake_lead_hours[self.fake_lead_indices]}h")
        else:
            raise ValueError(f"Fake dataset {fake_nc_path} is missing 'prediction_timedelta' dimension.")

        if means is None or stds is None:
            print(f"Calculating global normalization statistics from REAL dataset...")
            self.means = {var: float(self.real_ds[var].mean()) for var in variables}
            self.stds = {var: float(self.real_ds[var].std()) for var in variables}
            print(f"Global Stats - Means: {self.means}, Stds: {self.stds}")
        else:
            self.means = means
            self.stds = stds

        # Class balancing
        self.num_fake_leads = len(self.fake_lead_indices)
        self.total_fake_samples = len(self.fake_times) * self.num_fake_leads
        self.total_real_samples = len(self.real_times)
        self.samples_per_class = max(self.total_fake_samples, self.total_real_samples)

    def __len__(self):
        return self.samples_per_class * 2

    def __getitem__(self, idx):
        is_fake = idx >= self.samples_per_class
        sub_idx = idx % self.samples_per_class
        
        if not is_fake:
            # REAL
            ds_to_use = self.real_ds
            time_val = self.real_times[sub_idx % self.total_real_samples]
            lead_idx = self.real_lead_idx
            label = torch.tensor([1.0], dtype=torch.float32)
        else:
            # FAKE
            ds_to_use = self.fake_ds
            f_idx = sub_idx % self.total_fake_samples
            time_idx = f_idx // self.num_fake_leads
            lead_inner_idx = f_idx % self.num_fake_leads
            
            time_val = self.fake_times[time_idx]
            lead_idx = self.fake_lead_indices[lead_inner_idx]
            label = torch.tensor([0.0], dtype=torch.float32)

        if lead_idx is not None:
            ds_slice = ds_to_use.sel(time=time_val).isel(prediction_timedelta=lead_idx)
        else:
            ds_slice = ds_to_use.sel(time=time_val)

        channels = []
        for var in self.variables:
            data_array = ds_slice[var].values.astype(np.float32)
            mean, std = self.means[var], self.stds[var]
            norm_data = (data_array - mean) / std if std > 1e-8 else data_array - mean
            norm_data = np.nan_to_num(norm_data, nan=0.0)
            channels.append(norm_data)
            
        return torch.tensor(np.stack(channels), dtype=torch.float32), label

# ==========================================
# 2. Lightning Module Definition
# ==========================================
class WeatherDiscriminator(L.LightningModule):
    def __init__(self, num_weather_channels, model_name="resnet18", learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model_name = model_name
        
        if model_name == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            old_conv = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                num_weather_channels, old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
                padding=old_conv.padding, bias=old_conv.bias is not None
            )
            with torch.no_grad():
                repeat_factor = (num_weather_channels // 3) + 1
                new_weights = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :num_weather_channels, :, :]
                self.model.conv1.weight = nn.Parameter(new_weights * (3.0 / num_weather_channels))
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)
            
        elif model_name == "squeezenet":
            self.model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
            old_conv = self.model.features[0]
            self.model.features[0] = nn.Conv2d(
                num_weather_channels, old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, stride=old_conv.stride
            )
            with torch.no_grad():
                repeat_factor = (num_weather_channels // 3) + 1
                new_weights = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :num_weather_channels, :, :]
                self.model.features[0].weight = nn.Parameter(new_weights * (3.0 / num_weather_channels))
            
            old_classifier = self.model.classifier[1]
            self.model.classifier[1] = nn.Conv2d(old_classifier.in_channels, 1, kernel_size=(1, 1))
            self.model.classifier[2] = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        out = self.model(x)
        if self.model_name == "squeezenet":
            out = torch.flatten(out, 1)
        return out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = ((outputs > 0.0).float() == labels).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    selected_vars = [cfg.selected_variable]
    real_path, fake_path = cfg.real_nc_file, cfg.fake_nc_file
    
    dataset = WeatherDiscriminatorDataset(real_path, fake_path, selected_vars)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    model = WeatherDiscriminator(num_weather_channels=1, model_name=cfg.model_name, learning_rate=cfg.learning_rate)
    
    wandb_logger = WandbLogger(project=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True))
    trainer = L.Trainer(max_epochs=cfg.epochs, logger=wandb_logger, accelerator="auto", precision=cfg.precision if torch.cuda.is_available() else 32)
    
    trainer.fit(model, dataloader)

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, f"weather_discriminator_{cfg.model_name}_lightning.pth")
    torch.save(model.model.state_dict(), save_path)
    print(f"Weights saved to {save_path}")

if __name__ == "__main__":
    main()
