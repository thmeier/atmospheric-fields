import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# Import dataset from training script
from train_discriminator import WeatherDiscriminatorDataset

def compute_psd_components(data):
    """Computes the absolute components of the 2D FFT (PSD components)."""
    # Remove mean to focus on variations
    data = data - np.mean(data)
    # 2D FFT
    fft_data = np.fft.fft2(data)
    # Return absolute components (flattened)
    return np.abs(fft_data).flatten()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def plot_histograms(cfg: DictConfig):
    # --- Setup ---
    nc_file = cfg.nc_file
    variables = cfg.variables
    
    print(f"Loading data from {nc_file}...")
    dataset = WeatherDiscriminatorDataset(nc_file, variables)
    num_times = len(dataset.times)
    
    # We will collect components for Real and Fake samples
    # To keep it efficient, we might want to subsample if the dataset is huge, 
    # but for now let's try to process all or a large representative chunk.
    max_samples = min(100, num_times) 
    
    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 6 * len(variables)))
    if len(variables) == 1:
        axes = [axes]

    for v_idx, var in enumerate(variables):
        real_components = []
        fake_components = []
        
        print(f"Processing variable: {var}...")
        for i in tqdm(range(max_samples)):
            # Real sample
            real_sample, _ = dataset[i]
            real_data = real_sample[v_idx].numpy()
            real_components.append(compute_psd_components(real_data))
            
            # Fake sample
            fake_sample, _ = dataset[i + num_times]
            fake_data = fake_sample[v_idx].numpy()
            fake_components.append(compute_psd_components(fake_data))
        
        # Concatenate all components
        real_all = np.concatenate(real_components)
        fake_all = np.concatenate(fake_components)
        
        # Plot Histograms
        # We use log scale for the x-axis (frequencies) as PSD components usually span several orders
        # And log scale for y-axis to see the tails of the distribution
        ax = axes[v_idx]
        
        # Filter out zeros for log plotting
        real_all = real_all[real_all > 0]
        fake_all = fake_all[fake_all > 0]
        
        bins = np.logspace(np.log10(min(real_all.min(), fake_all.min())), 
                          np.log10(max(real_all.max(), fake_all.max())), 100)
        
        ax.hist(real_all, bins=bins, alpha=0.5, label='Real (Lead 0)', color='blue', density=True)
        ax.hist(fake_all, bins=bins, alpha=0.5, label='Fake (Lead 12h)', color='red', density=True)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"PSD Component Distribution: {var}")
        ax.set_xlabel("Absolute Magnitude of FFT Components")
        ax.set_ylabel("Density (Log Scale)")
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    save_path = "psd_histograms.png"
    plt.savefig(save_path, dpi=200)
    print(f"PSD Histograms saved to {save_path}")

if __name__ == "__main__":
    plot_histograms()
