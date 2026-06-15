"""Shared helpers for discriminator evaluation and analysis scripts.

The training script owns the model and training dataset.  This module keeps the
post-training scripts small by centralizing the repeated inference-only pieces:
time-range selection, ERA5 normalization statistics, and lead-time batching.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .train_discriminator import WeatherDiscriminator, select_time_ranges
except ImportError:
    from train_discriminator import WeatherDiscriminator, select_time_ranges


def resolve_device():
    """Use CUDA when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_weight_path(cfg, variables, *, filename=None):
    """Return the default `.pth` path produced by `train_discriminator.py`."""
    if filename is not None:
        return Path(cfg.output_dir) / filename

    variable_tag = cfg.get("selected_variable", "all_fields") if len(variables) == 1 else "all_fields"
    return Path(cfg.output_dir) / f"weather_discriminator_{cfg.model_name}_{variable_tag}_lightning.pth"


def load_discriminator(cfg, variables, device, *, filename=None):
    """Construct a discriminator and load the saved inner torchvision weights."""
    path = model_weight_path(cfg, variables, filename=filename)
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path}")

    model = WeatherDiscriminator(len(variables), cfg.model_name).to(device)
    model.model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, path


def normalization_stats(real_ds, variables, train_ranges):
    """Compute the same z-score statistics used for model inputs."""
    stats_ds = select_time_ranges(real_ds, train_ranges)
    means = {}
    stds = {}

    for variable in variables:
        if variable not in stats_ds.data_vars:
            means[variable] = 0.0
            stds[variable] = 1.0
            continue

        means[variable] = float(stats_ds[variable].mean())
        std = float(stats_ds[variable].std())
        stds[variable] = std if std > 1e-8 else 1.0

    return means, stds


def normalized_channels(ds_slice, variables, means, stds):
    """Convert an xarray slice into a normalized CxHxW torch tensor."""
    channels = []
    for variable in variables:
        if variable in ds_slice.data_vars:
            raw = ds_slice[variable].values.astype(np.float32)
            normalized = (raw - means[variable]) / stds[variable]
            channels.append(np.nan_to_num(normalized, nan=0.0))
            continue

        # Missing variables are explicit zero channels.  Most current analysis
        # paths check for missing variables earlier, but this keeps old files
        # readable when one field is absent from a comparison dataset.
        reference = next(iter(ds_slice.data_vars.values()))
        channels.append(np.zeros(reference.shape, dtype=np.float32))

    return torch.tensor(np.stack(channels), dtype=torch.float32)


def to_int_lead_hours(values):
    """Normalize xarray lead-time coordinates to integer hours."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[h]").astype(int)
    return values.astype(int)


class LeadTimeInferenceDataset(Dataset):
    """Inference dataset that returns `(sample, lead_index)` for every lead."""

    def __init__(self, ds, variables, means, stds, level=None, max_samples=0):
        self.ds = ds
        if level is not None:
            if "level" in self.ds.dims:
                self.ds = self.ds.sel(level=level)
            elif "pressure_level" in self.ds.dims:
                self.ds = self.ds.sel(pressure_level=level)

        self.variables = list(variables)
        self.means = means
        self.stds = stds

        if max_samples > 0 and self.ds.sizes.get("time", 0) > max_samples:
            indices = np.linspace(0, self.ds.sizes["time"] - 1, max_samples, dtype=int)
            self.ds = self.ds.isel(time=indices)

        if "prediction_timedelta" in self.ds.dims:
            self.lead_hours = to_int_lead_hours(self.ds.prediction_timedelta.values)
            self.num_leads = len(self.lead_hours)
            self.has_lead_dim = True
        else:
            self.lead_hours = np.array([0])
            self.num_leads = 1
            self.has_lead_dim = False

    def __len__(self):
        return self.ds.sizes.get("time", 0) * self.num_leads

    def __getitem__(self, idx):
        time_idx = idx // self.num_leads
        lead_idx = idx % self.num_leads

        ds_slice = self.ds.isel(time=time_idx)
        if self.has_lead_dim:
            ds_slice = ds_slice.isel(prediction_timedelta=lead_idx)

        return normalized_channels(ds_slice, self.variables, self.means, self.stds), lead_idx


def mean_logits_by_lead(dataset, model, cfg, device, desc="Inference"):
    """Run model inference and summarize logits independently per lead time."""
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    logits_by_lead = [[] for _ in range(dataset.num_leads)]

    model.eval()
    with torch.no_grad():
        for inputs, lead_indices in tqdm(loader, desc=desc, leave=False):
            logits = model(inputs.to(device)).cpu().numpy().flatten()
            for logit, lead_idx in zip(logits, lead_indices):
                logits_by_lead[int(lead_idx)].append(float(logit))

    means = [float(np.mean(values)) for values in logits_by_lead]
    stds = [float(np.std(values)) for values in logits_by_lead]
    return dataset.lead_hours, means, stds
