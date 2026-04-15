import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from scipy import signal

# Import dataset from training script
from train_discriminator import WeatherDiscriminatorDataset

def compute_2d_psd(data):
    """Computes the 2D Power Spectral Density of a 2D array."""
    # Remove mean to focus on variations
    data = data - np.mean(data)
    # 2D FFT
    fft_data = np.fft.fft2(data)
    fft_shifted = np.fft.fftshift(fft_data)
    # Power Spectrum
    psd = np.abs(fft_shifted)**2
    # Log scale for visualization
    return np.log10(psd + 1e-8)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def plot_spectrograms(cfg: DictConfig):
    # --- Setup ---
    nc_file = cfg.nc_file
    variables = cfg.variables
    
    print(f"Loading data from {nc_file}...")
    dataset = WeatherDiscriminatorDataset(nc_file, variables)
    
    # We want to compare Lead Time 0 (Real) vs Lead Time 12 (Fake)
    # The dataset is structured as: [0...N-1] are Real, [N...2N-1] are Fake
    num_times = len(dataset.times)
    
    # Select a random time index to showcase
    time_idx = num_times // 2 
    
    real_sample, _ = dataset[time_idx]             # Real (Lead 0)
    fake_sample, _ = dataset[time_idx + num_times] # Fake (Lead 12)

    fig, axes = plt.subplots(len(variables), 4, figsize=(20, 5 * len(variables)))
    if len(variables) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, var in enumerate(variables):
        real_data = real_sample[i].numpy()
        fake_data = fake_sample[i].numpy()
        
        # 1. Plot Spatial Real
        axes[i, 0].imshow(real_data, cmap='viridis')
        axes[i, 0].set_title(f"Real: {var}\n(Lead 0)")
        axes[i, 0].axis('off')
        
        # 2. Plot Spectrogram Real
        real_psd = compute_2d_psd(real_data)
        im1 = axes[i, 1].imshow(real_psd, cmap='magma')
        axes[i, 1].set_title(f"PSD (Log10): {var}\nReal")
        fig.colorbar(im1, ax=axes[i, 1])
        
        # 3. Plot Spatial Fake
        axes[i, 2].imshow(fake_data, cmap='viridis')
        axes[i, 2].set_title(f"Fake: {var}\n(Lead 12h)")
        axes[i, 2].axis('off')
        
        # 4. Plot Spectrogram Fake
        fake_psd = compute_2d_psd(fake_data)
        im2 = axes[i, 3].imshow(fake_psd, cmap='magma')
        axes[i, 3].set_title(f"PSD (Log10): {var}\nFake")
        fig.colorbar(im2, ax=axes[i, 3])

    plt.tight_layout()
    save_path = "field_spectrograms.png"
    plt.savefig(save_path, dpi=200)
    print(f"Spectrogram plot saved to {save_path}")

if __name__ == "__main__":
    plot_spectrograms()
