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
    def __init__(self, real_nc_path, fake_nc_path, variables, 
                 real_range=None, fake_range=None, lead_times=None, level=None, 
                 means=None, stds=None, balanced=True):
        self.variables = variables
        self.balanced = balanced
        self.real_ds = xr.open_dataset(real_nc_path)
        self.fake_ds = xr.open_dataset(fake_nc_path)
        
        # 1. Level Selection
        if level is not None:
            for ds_attr in ['real_ds', 'fake_ds']:
                ds = getattr(self, ds_attr)
                if 'level' in ds.dims:
                    setattr(self, ds_attr, ds.sel(level=level))
                elif 'pressure_level' in ds.dims:
                    setattr(self, ds_attr, ds.sel(pressure_level=level))

        # 2. Time Range Selection
        if real_range:
            # Check if it's a list of ranges (nested list/ListConfig)
            # If the first element is not a string, we assume it's a [start, end] pair
            if not isinstance(real_range[0], str):
                combined = [self.real_ds.sel(time=slice(r[0], r[1])) for r in real_range]
                self.real_ds = xr.concat(combined, dim='time')
            else:
                self.real_ds = self.real_ds.sel(time=slice(real_range[0], real_range[1]))
        
        if fake_range:
            self.fake_ds = self.fake_ds.sel(time=slice(fake_range[0], fake_range[1]))
            
        self.real_times = self.real_ds.time.values
        self.fake_times = self.fake_ds.time.values
        
        # 3. Lead Time Filtering
        if "prediction_timedelta" in self.fake_ds.dims:
            lt = self.fake_ds.prediction_timedelta.values
            lt_h = lt.astype('timedelta64[h]').astype(int) if np.issubdtype(lt.dtype, np.timedelta64) else lt
            if lead_times is not None:
                self.fake_lead_indices = [np.where(lt_h == h)[0][0] for h in lead_times if h in lt_h]
            else:
                self.fake_lead_indices = np.where(lt_h > 0)[0].tolist()
        else:
            self.fake_lead_indices = [None]

        if "prediction_timedelta" in self.real_ds.dims:
            lt = self.real_ds.prediction_timedelta.values
            lt_h = lt.astype('timedelta64[h]').astype(int) if np.issubdtype(lt.dtype, np.timedelta64) else lt
            zero_idx = np.where(lt_h == 0)[0]
            self.real_lead_idx = int(zero_idx[0]) if len(zero_idx) > 0 else 0
        else:
            self.real_lead_idx = None

        # 4. Normalization Stats (from REAL)
        if means is None or stds is None:
            print(f"Calculating stats from REAL dataset...")
            self.means = {}
            self.stds = {}
            for v in self.variables:
                if v in self.real_ds.data_vars:
                    self.means[v] = float(self.real_ds[v].mean())
                    self.stds[v] = float(self.real_ds[v].std())
                else:
                    self.means[v] = 0.0
                    self.stds[v] = 1.0
        else:
            self.means, self.stds = means, stds

        # 5. Class Balancing
        self.num_fake_leads = len(self.fake_lead_indices)
        self.total_fake_samples = len(self.fake_times) * self.num_fake_leads
        self.total_real_samples = len(self.real_times)

        
        if self.balanced:
            self.samples_per_class = max(self.total_fake_samples, self.total_real_samples)
        
        print(f"Dataset Initialized (Balanced={self.balanced}):")
        print(f"  REAL: {self.total_real_samples} | FAKE: {self.total_fake_samples}")

    def __len__(self):
        if self.balanced:
            return self.samples_per_class * 2
        else:
            return self.total_real_samples + self.total_fake_samples

    def __getitem__(self, idx):
        if self.balanced:
            is_fake = idx >= self.samples_per_class
            sub_idx = idx % self.samples_per_class
            if not is_fake:
                ds, t_val, l_idx = self.real_ds, self.real_times[sub_idx % self.total_real_samples], self.real_lead_idx
                label = torch.tensor([1.0], dtype=torch.float32)
            else:
                f_idx = sub_idx % self.total_fake_samples
                ds, t_val, l_idx = self.fake_ds, self.fake_times[f_idx // self.num_fake_leads], self.fake_lead_indices[f_idx % self.num_fake_leads]
                label = torch.tensor([0.0], dtype=torch.float32)
        else:
            is_fake = idx >= self.total_real_samples
            if not is_fake:
                ds, t_val, l_idx = self.real_ds, self.real_times[idx], self.real_lead_idx
                label = torch.tensor([1.0], dtype=torch.float32)
            else:
                f_idx = idx - self.total_real_samples
                ds, t_val, l_idx = self.fake_ds, self.fake_times[f_idx // self.num_fake_leads], self.fake_lead_indices[f_idx % self.num_fake_leads]
                label = torch.tensor([0.0], dtype=torch.float32)

        ds_slice = ds.sel(time=t_val)
        if l_idx is not None:
            ds_slice = ds_slice.isel(prediction_timedelta=l_idx)

        channels = []
        for v in self.variables:
            if v in ds_slice.data_vars:
                val = ds_slice[v].values.astype(np.float32)
                norm = (val - self.means[v]) / (self.stds[v] if self.stds[v] > 1e-8 else 1.0)
                channels.append(np.nan_to_num(norm, nan=0.0))
            else:
                # Zero-fill missing variables to maintain channel count
                # Using temperature as a reference for shape
                ref_shape = ds_slice.temperature.shape if 'temperature' in ds_slice.data_vars else list(ds_slice.data_vars.values())[0].shape
                channels.append(np.zeros(ref_shape, dtype=np.float32))
            
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
    vars_to_use = list(cfg.variables)
    real_path, fake_path = cfg.real_nc_file, cfg.fake_nc_file
    
    dataset = WeatherDiscriminatorDataset(
        real_path, fake_path, vars_to_use,
        real_range=cfg.train_real_range,
        fake_range=cfg.train_fake_range,
        lead_times=cfg.lead_times,
        level=cfg.get("level"),
        balanced=True
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    model = WeatherDiscriminator(
        num_weather_channels=len(vars_to_use), 
        model_name=cfg.model_name, 
        learning_rate=cfg.learning_rate
    )
    
    wandb_logger = WandbLogger(project=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True))
    trainer = L.Trainer(max_epochs=cfg.epochs, logger=wandb_logger, accelerator="auto", precision=cfg.precision if torch.cuda.is_available() else 32)
    
    trainer.fit(model, dataloader)

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, f"weather_discriminator_{cfg.model_name}_lightning.pth")
    torch.save(model.model.state_dict(), save_path)
    print(f"Weights saved to {save_path}")

if __name__ == "__main__":
    main()
