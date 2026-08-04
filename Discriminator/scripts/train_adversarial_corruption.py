"""Search for coherent ERA5 perturbations missed by standard distribution metrics.

This is deliberately a cooperative, rather than conventional adversarial, game:
the residual generator is rewarded for producing perturbations that a learned
discriminator can identify while differentiable SCWD and spectral surrogates
remain inside their clean-vs-clean minibatch variability.
"""

import csv
import math
import tempfile
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .plot_standard_metric_baselines import (
        scwd_weight_vectors,
        zonal_energy_spectrum_log_l2,
    )
    from .train_discriminator import (
        WeatherDiscriminator,
        safe_open_dataset,
        select_time_ranges,
        validate_no_train_test_overlap,
    )
except ImportError:
    from plot_standard_metric_baselines import (
        scwd_weight_vectors,
        zonal_energy_spectrum_log_l2,
    )
    from train_discriminator import (
        WeatherDiscriminator,
        safe_open_dataset,
        select_time_ranges,
        validate_no_train_test_overlap,
    )


def experiment_get(cfg, key, default=None):
    section = cfg.get("adversarial_corruption")
    if section is None:
        raise ValueError("Missing adversarial_corruption configuration section.")
    value = section.get(key)
    return default if value is None else value


def seed_everything(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(cfg):
    configured = str(experiment_get(cfg, "device", "auto"))
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Configured device {device} is unavailable.")
    return device


def cosine_latitude_weights(latitudes, device=None, dtype=torch.float32):
    values = torch.as_tensor(np.cos(np.deg2rad(np.asarray(latitudes))), dtype=dtype, device=device)
    return torch.clamp(values, min=0.0)


class ERA5TemperatureDataset(Dataset):
    """Lazy, train-normalized scalar ERA5 fields from configured time ranges."""

    def __init__(self, dataset, variable, ranges, mean, std, max_samples=0):
        selected = select_time_ranges(dataset, ranges)
        if selected.sizes.get("time", 0) == 0:
            raise ValueError(f"No ERA5 samples found for ranges={ranges}.")
        if variable not in selected.data_vars:
            raise ValueError(f"ERA5 does not contain configured variable {variable}.")
        if max_samples and selected.sizes["time"] > int(max_samples):
            indices = np.linspace(0, selected.sizes["time"] - 1, int(max_samples), dtype=int)
            selected = selected.isel(time=indices)
        self.dataset = selected
        self.variable = str(variable)
        self.mean = float(mean)
        self.std = max(float(std), 1e-8)
        self.latitudes = np.asarray(selected.latitude.values, dtype=np.float64)
        self.longitudes = np.asarray(selected.longitude.values, dtype=np.float64)

    def __len__(self):
        return self.dataset.sizes["time"]

    def __getitem__(self, index):
        field = self.dataset[self.variable].isel(time=int(index)).transpose(
            "latitude", "longitude"
        ).values.astype(np.float32)
        normalized = np.nan_to_num((field - self.mean) / self.std)
        return torch.from_numpy(normalized[None, ...])

    def time_value(self, index):
        return self.dataset.time.values[int(index)]


def training_normalization(dataset, variable, ranges):
    selected = select_time_ranges(dataset, ranges)
    if selected.sizes.get("time", 0) == 0:
        raise ValueError("No ERA5 training samples available for normalization.")
    values = selected[variable]
    mean = float(values.mean())
    std = float(values.std())
    return mean, max(std, 1e-8)


class SphericalConv2d(nn.Module):
    """Convolution with circular longitude and reflected latitude padding."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.padding = int(kernel_size) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, values):
        pad = self.padding
        values = functional.pad(values, (pad, pad, 0, 0), mode="circular")
        values = functional.pad(values, (0, 0, pad, pad), mode="reflect")
        return self.conv(values)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = max(1, min(8, out_channels // 4))
        while out_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            SphericalConv2d(in_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            SphericalConv2d(out_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, values):
        return self.layers(values)


def periodic_bilinear_upsample(values, target_shape):
    """Bilinearly resize while extending longitude periodically across the seam."""
    target_h, target_w = (int(target_shape[0]), int(target_shape[1]))
    coarse_w = values.shape[-1]
    margin = max(1, int(math.ceil(target_w / coarse_w)))
    extended = torch.cat((values[..., -1:], values, values[..., :1]), dim=-1)
    resized = functional.interpolate(
        extended,
        size=(target_h, target_w + 2 * margin),
        mode="bilinear",
        align_corners=False,
    )
    return resized[..., margin:margin + target_w]


def residual_statistics(residual, latitude_weights):
    weights = latitude_weights[None, None, :, None]
    normalizer = torch.sum(latitude_weights) * residual.shape[-1]
    energy = residual.square()
    weighted_energy = torch.sum(weights * energy, dim=(-2, -1))
    rms = torch.sqrt(weighted_energy / normalizer + 1e-12)
    fourth = torch.sum(weights * energy.square(), dim=(-2, -1))
    effective_fraction = weighted_energy.square() / (normalizer * fourth + 1e-12)
    peak_ratio = torch.amax(torch.abs(residual), dim=(-2, -1)) / (rms + 1e-12)
    return rms, effective_fraction, peak_ratio


def constrain_residual(raw_residual, latitude_weights, target_rms):
    """Remove global mean and project each sample to fixed area-weighted RMS."""
    weights = latitude_weights[None, None, :, None]
    normalizer = torch.sum(latitude_weights) * raw_residual.shape[-1]
    weighted_mean = torch.sum(weights * raw_residual, dim=(-2, -1), keepdim=True) / normalizer
    centered = raw_residual - weighted_mean
    rms = torch.sqrt(
        torch.sum(weights * centered.square(), dim=(-2, -1), keepdim=True) / normalizer
        + 1e-12
    )
    return centered * (float(target_rms) / rms)


class CoarseResidualUNet(nn.Module):
    """Small U-Net producing a smooth, fixed-energy residual field."""

    def __init__(self, base_channels=16, coarsening=4):
        super().__init__()
        width = int(base_channels)
        self.coarsening = int(coarsening)
        self.encoder1 = ConvBlock(1, width)
        self.encoder2 = ConvBlock(width, 2 * width)
        self.bottleneck = ConvBlock(2 * width, 4 * width)
        self.decoder2 = ConvBlock(6 * width, 2 * width)
        self.decoder1 = ConvBlock(3 * width, width)
        self.output = SphericalConv2d(width, 1, kernel_size=1)

    def raw_residual(self, inputs):
        original_shape = inputs.shape[-2:]
        coarse_shape = (
            max(4, int(math.ceil(original_shape[0] / self.coarsening))),
            max(4, int(math.ceil(original_shape[1] / self.coarsening))),
        )
        coarse = functional.interpolate(inputs, size=coarse_shape, mode="area")
        enc1 = self.encoder1(coarse)
        enc2 = self.encoder2(functional.avg_pool2d(enc1, 2, ceil_mode=True))
        bottleneck = self.bottleneck(functional.avg_pool2d(enc2, 2, ceil_mode=True))
        up2 = functional.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.decoder2(torch.cat((up2, enc2), dim=1))
        up1 = functional.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.decoder1(torch.cat((up1, enc1), dim=1))
        return periodic_bilinear_upsample(self.output(dec1), original_shape)

    def forward(self, inputs, latitude_weights, target_rms):
        residual = constrain_residual(self.raw_residual(inputs), latitude_weights, target_rms)
        return inputs + residual, residual


def effective_area_penalty(residual, latitude_weights, minimum_fraction):
    _, fractions, _ = residual_statistics(residual, latitude_weights)
    return functional.relu(float(minimum_fraction) - fractions).square().mean()


def peak_penalty(residual, target_rms, maximum_ratio):
    excess = functional.relu(torch.abs(residual) - float(maximum_ratio) * float(target_rms))
    return torch.mean((excess / max(float(target_rms), 1e-8)) ** 2)


def differentiable_zonal_log_spectrum_distance(candidate, reference, latitude_weights, eps_factor=1e-12):
    """Torch counterpart of the canonical mean zonal log-spectrum L2."""
    weights = latitude_weights[None, None, :, None]
    weight_sum = torch.sum(latitude_weights)
    candidate_power = torch.abs(torch.fft.rfft(candidate, dim=-1)) ** 2
    reference_power = torch.abs(torch.fft.rfft(reference, dim=-1)) ** 2
    candidate_spectrum = torch.sum(weights * candidate_power, dim=2) / weight_sum
    reference_spectrum = torch.sum(weights * reference_power, dim=2) / weight_sum
    candidate_spectrum = torch.mean(candidate_spectrum, dim=(0, 1))
    reference_spectrum = torch.mean(reference_spectrum, dim=(0, 1))
    reference_scale = torch.clamp(torch.max(reference_spectrum.detach()), min=1e-12)
    epsilon = reference_scale * float(eps_factor)
    difference = torch.log(candidate_spectrum + epsilon) - torch.log(reference_spectrum + epsilon)
    return torch.sqrt(torch.mean(difference.square()) + 1e-12)


def sparse_anchor_matrix(weights, anchor_indices, n_pixels, device):
    rows, columns, values = [], [], []
    for output_row, anchor_index in enumerate(np.asarray(anchor_indices, dtype=int)):
        support, support_weights = weights[int(anchor_index)]
        if not support.size:
            continue
        rows.append(np.full(support.size, output_row, dtype=np.int64))
        columns.append(np.asarray(support, dtype=np.int64))
        values.append(np.asarray(support_weights, dtype=np.float32))
    if not rows:
        raise ValueError("Selected SCWD anchor bank has no supported filters.")
    indices = torch.as_tensor(
        np.stack((np.concatenate(rows), np.concatenate(columns))),
        dtype=torch.long,
        device=device,
    )
    data = torch.as_tensor(np.concatenate(values), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices,
        data,
        size=(len(anchor_indices), int(n_pixels)),
        device=device,
        check_invariants=False,
    ).coalesce()


def build_anchor_banks(cfg, latitudes, longitudes, device):
    reference = {"latitudes": latitudes, "longitudes": longitudes}
    weights = scwd_weight_vectors(reference, cfg)
    n_anchors = len(weights)
    bank_size = min(int(experiment_get(cfg, "training_scwd_anchors", 512)), n_anchors)
    n_banks = max(1, int(experiment_get(cfg, "training_scwd_anchor_banks", 16)))
    rng = np.random.default_rng(int(experiment_get(cfg, "seed", 0)))
    matrices = []
    for _ in range(n_banks):
        indices = rng.choice(n_anchors, size=bank_size, replace=False)
        matrices.append(sparse_anchor_matrix(weights, indices, len(latitudes) * len(longitudes), device))
    full_matrix = sparse_anchor_matrix(weights, np.arange(n_anchors), len(latitudes) * len(longitudes), device)
    return matrices, full_matrix


def differentiable_scwd(candidate, reference, weight_matrix, n_quantiles=200, order=2.0):
    candidate_flat = candidate[:, 0].reshape(candidate.shape[0], -1)
    reference_flat = reference[:, 0].reshape(reference.shape[0], -1)
    candidate_response = torch.sparse.mm(weight_matrix, candidate_flat.T).T
    reference_response = torch.sparse.mm(weight_matrix, reference_flat.T).T
    quantiles = torch.linspace(0.0, 1.0, int(n_quantiles), device=candidate.device)
    candidate_quantiles = torch.quantile(candidate_response, quantiles, dim=0)
    reference_quantiles = torch.quantile(reference_response, quantiles, dim=0)
    return torch.mean(torch.abs(candidate_quantiles - reference_quantiles) ** float(order)) ** (
        1.0 / float(order)
    )


def independent_loaders(dataset, batch_size, num_workers, seed, drop_last=True):
    first_generator = torch.Generator().manual_seed(int(seed))
    second_generator = torch.Generator().manual_seed(int(seed) + 104729)
    common = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": True,
        "num_workers": int(num_workers),
        "drop_last": bool(drop_last and len(dataset) >= int(batch_size)),
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(generator=first_generator, **common),
        DataLoader(generator=second_generator, **common),
    )


def random_longitude_roll(values, coarsening, rng):
    possible = max(1, values.shape[-1] // int(coarsening))
    shift = int(rng.integers(0, possible)) * int(coarsening)
    return torch.roll(values, shifts=shift, dims=-1)


def calibrate_metric_thresholds(cfg, dataset, anchor_banks, latitude_weights, device):
    """Estimate train-only clean-vs-clean minibatch metric variability."""
    batch_size = int(experiment_get(cfg, "batch_size", 32))
    loader_a, loader_b = independent_loaders(
        dataset,
        batch_size,
        experiment_get(cfg, "num_workers", 0),
        experiment_get(cfg, "seed", 0),
    )
    requested = int(experiment_get(cfg, "null_calibration_pairs", 128))
    scwd_values, spectrum_values = [], []
    with torch.no_grad():
        for pair_index, (clean_a, clean_b) in enumerate(zip(loader_a, loader_b)):
            if pair_index >= requested:
                break
            clean_a = clean_a.to(device)
            clean_b = clean_b.to(device)
            bank = anchor_banks[pair_index % len(anchor_banks)]
            scwd_values.append(float(differentiable_scwd(clean_a, clean_b, bank).cpu()))
            spectrum_values.append(
                float(differentiable_zonal_log_spectrum_distance(clean_a, clean_b, latitude_weights).cpu())
            )
    if not scwd_values:
        raise ValueError("Metric null calibration produced no batch pairs.")
    quantile = float(experiment_get(cfg, "null_quantile", 0.95))
    return {
        "scwd": max(float(np.quantile(scwd_values, quantile)), 1e-8),
        "spectrum": max(float(np.quantile(spectrum_values, quantile)), 1e-8),
        "pairs": len(scwd_values),
    }


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def discriminator_loss(discriminator, clean, fake):
    clean_logits = discriminator(clean)
    fake_logits = discriminator(fake)
    loss = functional.binary_cross_entropy_with_logits(clean_logits, torch.ones_like(clean_logits))
    loss += functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    accuracy = torch.mean(
        torch.cat(((clean_logits > 0).float(), (fake_logits <= 0).float()), dim=0)
    )
    return loss, accuracy, clean_logits, fake_logits


def generator_loss(
    cfg,
    discriminator,
    fake,
    reference,
    residual,
    latitude_weights,
    anchor_matrix,
    thresholds,
    target_rms,
):
    scwd = differentiable_scwd(
        fake,
        reference,
        anchor_matrix,
        n_quantiles=experiment_get(cfg, "scwd_quantiles", 200),
        order=experiment_get(cfg, "scwd_order", 2.0),
    )
    spectrum = differentiable_zonal_log_spectrum_distance(
        fake,
        reference,
        latitude_weights,
        experiment_get(cfg, "spectrum_log_eps_factor", 1e-12),
    )
    metric_loss = functional.relu(scwd / thresholds["scwd"] - 1.0) + functional.relu(
        spectrum / thresholds["spectrum"] - 1.0
    )
    fake_logits = discriminator(fake)
    detectability = functional.relu(
        torch.mean(fake_logits) + float(experiment_get(cfg, "fake_logit_margin", 2.0))
    )
    peak = peak_penalty(residual, target_rms, experiment_get(cfg, "maximum_peak_ratio", 3.0))
    spread = effective_area_penalty(
        residual, latitude_weights, experiment_get(cfg, "minimum_effective_area_fraction", 0.2)
    )
    total = (
        float(experiment_get(cfg, "metric_loss_weight", 1.0)) * metric_loss
        + float(experiment_get(cfg, "detectability_weight", 1.0)) * detectability
        + float(experiment_get(cfg, "constraint_weight", 10.0)) * (peak + spread)
    )
    return total, {
        "generator_loss": total,
        "scwd": scwd,
        "spectrum": spectrum,
        "metric_hinge": metric_loss,
        "detectability": detectability,
        "peak_penalty": peak,
        "spread_penalty": spread,
        "fake_logit": torch.mean(fake_logits),
    }


def mean_dict(rows):
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def train_joint_pair(cfg, dataset, latitude_weights, anchor_banks, target_rms, device, output_dir):
    generator = CoarseResidualUNet(
        experiment_get(cfg, "generator_base_channels", 16),
        experiment_get(cfg, "generator_coarsening", 4),
    ).to(device)
    discriminator = WeatherDiscriminator(
        1,
        model_name="squeezenet",
        learning_rate=experiment_get(cfg, "learning_rate", 1e-4),
        pretrained_backbone=bool(experiment_get(cfg, "pretrained_discriminator", True)),
    ).to(device)
    generator_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=float(experiment_get(cfg, "learning_rate", 1e-4)),
        weight_decay=float(experiment_get(cfg, "weight_decay", 1e-4)),
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(experiment_get(cfg, "learning_rate", 1e-4)),
        weight_decay=float(experiment_get(cfg, "weight_decay", 1e-4)),
    )
    thresholds = calibrate_metric_thresholds(cfg, dataset, anchor_banks, latitude_weights, device)
    loader_a, loader_b = independent_loaders(
        dataset,
        experiment_get(cfg, "batch_size", 32),
        experiment_get(cfg, "num_workers", 0),
        int(experiment_get(cfg, "seed", 0)) + int(round(target_rms * 10000)),
    )
    rng = np.random.default_rng(int(experiment_get(cfg, "seed", 0)))
    history = []
    for epoch in range(int(experiment_get(cfg, "joint_epochs", 10))):
        epoch_rows = []
        progress = tqdm(zip(loader_a, loader_b), total=min(len(loader_a), len(loader_b)), desc=f"RMS {target_rms:.2f} epoch {epoch + 1}")
        for step, (clean, reference) in enumerate(progress):
            clean = random_longitude_roll(
                clean.to(device), experiment_get(cfg, "generator_coarsening", 4), rng
            )
            reference = random_longitude_roll(
                reference.to(device), experiment_get(cfg, "generator_coarsening", 4), rng
            )

            discriminator.train()
            generator.train()
            set_requires_grad(discriminator, True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_detached, _ = generator(clean, latitude_weights, target_rms)
            d_loss, d_accuracy, _, _ = discriminator_loss(discriminator, clean, fake_detached)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            discriminator_optimizer.step()

            discriminator.eval()
            set_requires_grad(discriminator, False)
            generator_optimizer.zero_grad(set_to_none=True)
            fake, residual = generator(clean, latitude_weights, target_rms)
            bank = anchor_banks[(epoch * len(loader_a) + step) % len(anchor_banks)]
            g_loss, parts = generator_loss(
                cfg,
                discriminator,
                fake,
                reference,
                residual,
                latitude_weights,
                bank,
                thresholds,
                target_rms,
            )
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            generator_optimizer.step()
            set_requires_grad(discriminator, True)

            rms, area, peak = residual_statistics(residual.detach(), latitude_weights)
            row = {key: float(value.detach().cpu()) for key, value in parts.items()}
            row.update(
                {
                    "discriminator_loss": float(d_loss.detach().cpu()),
                    "discriminator_accuracy": float(d_accuracy.detach().cpu()),
                    "residual_rms": float(rms.mean().cpu()),
                    "effective_area_fraction": float(area.mean().cpu()),
                    "peak_ratio": float(peak.mean().cpu()),
                }
            )
            epoch_rows.append(row)
            progress.set_postfix(g=f"{row['generator_loss']:.3f}", d=f"{row['discriminator_accuracy']:.2f}")
        summary = mean_dict(epoch_rows)
        summary.update({"epoch": epoch + 1, "target_rms": target_rms})
        history.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), output_dir / "generator.pth")
    torch.save(discriminator.model.state_dict(), output_dir / "joint_discriminator.pth")
    write_rows(history, output_dir / "training_history.csv")
    return generator, discriminator, thresholds, history


def train_fresh_discriminator(cfg, dataset, generator, latitude_weights, target_rms, device, output_dir):
    discriminator = WeatherDiscriminator(
        1,
        model_name="squeezenet",
        learning_rate=experiment_get(cfg, "learning_rate", 1e-4),
        pretrained_backbone=bool(experiment_get(cfg, "pretrained_discriminator", True)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(experiment_get(cfg, "learning_rate", 1e-4)),
        weight_decay=float(experiment_get(cfg, "weight_decay", 1e-4)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(experiment_get(cfg, "batch_size", 32)),
        shuffle=True,
        num_workers=int(experiment_get(cfg, "num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    generator.eval()
    for epoch in range(int(experiment_get(cfg, "fresh_discriminator_epochs", 10))):
        progress = tqdm(loader, desc=f"Fresh D RMS {target_rms:.2f} epoch {epoch + 1}")
        for clean in progress:
            clean = clean.to(device)
            with torch.no_grad():
                fake, _ = generator(clean, latitude_weights, target_rms)
            discriminator.train()
            optimizer.zero_grad(set_to_none=True)
            loss, accuracy, _, _ = discriminator_loss(discriminator, clean, fake)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            optimizer.step()
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.3f}", acc=f"{float(accuracy.detach().cpu()):.2f}")
    torch.save(discriminator.model.state_dict(), output_dir / "fresh_discriminator.pth")
    return discriminator


def random_coarse_perturbation(clean, latitude_weights, target_rms, coarsening, generator):
    coarse_shape = (
        max(4, int(math.ceil(clean.shape[-2] / int(coarsening)))),
        max(4, int(math.ceil(clean.shape[-1] / int(coarsening)))),
    )
    raw = torch.randn(
        (clean.shape[0], 1, *coarse_shape),
        dtype=clean.dtype,
        device=clean.device,
        generator=generator,
    )
    raw = periodic_bilinear_upsample(raw, clean.shape[-2:])
    residual = constrain_residual(raw, latitude_weights, target_rms)
    return clean + residual, residual


def binary_auroc(real_scores, fake_scores):
    labels = np.concatenate((np.ones(len(real_scores)), np.zeros(len(fake_scores))))
    scores = np.concatenate((np.asarray(real_scores), np.asarray(fake_scores)))
    if not len(real_scores) or not len(fake_scores):
        return np.nan
    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    threshold_indices = np.concatenate(
        (np.flatnonzero(np.diff(sorted_scores)) + 1, [len(sorted_scores)])
    )
    cumulative_positive = np.cumsum(sorted_labels)
    cumulative_negative = np.cumsum(1.0 - sorted_labels)
    true_positive = np.concatenate(
        ([0.0], cumulative_positive[threshold_indices - 1] / len(real_scores))
    )
    false_positive = np.concatenate(
        ([0.0], cumulative_negative[threshold_indices - 1] / len(fake_scores))
    )
    return float(np.trapezoid(true_positive, false_positive))


def canonical_scwd(candidate, reference, order=2.0, n_quantiles=200, anchor_chunk=128):
    quantiles = np.linspace(0.0, 1.0, int(n_quantiles))
    n_anchors = min(candidate.shape[1], reference.shape[1])
    total = 0.0
    count = 0
    for start in range(0, n_anchors, int(anchor_chunk)):
        stop = min(start + int(anchor_chunk), n_anchors)
        candidate_q = np.quantile(candidate[:, start:stop], quantiles, axis=0)
        reference_q = np.quantile(reference[:, start:stop], quantiles, axis=0)
        total += float(np.sum(np.abs(candidate_q - reference_q) ** float(order)))
        count += (stop - start) * len(quantiles)
    return float((total / count) ** (1.0 / float(order))) if count else np.nan


def mean_zonal_spectrum(values, latitude_weights):
    return torch.sum(zonal_spectrum_per_sample(values, latitude_weights), dim=0).detach().cpu().numpy()


def zonal_spectrum_per_sample(values, latitude_weights):
    power = torch.abs(torch.fft.rfft(values, dim=-1)) ** 2
    spectrum = torch.sum(latitude_weights[None, None, :, None] * power, dim=2)
    spectrum = spectrum / torch.sum(latitude_weights)
    return torch.mean(spectrum, dim=1)


def relative_spectrum_l2(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    common = min(candidate.size, reference.size)
    if not common:
        return np.nan
    return float(
        np.linalg.norm(candidate[:common] - reference[:common])
        / (np.linalg.norm(reference[:common]) + 1e-12)
    )


def discriminator_summary(real_scores, fake_scores):
    real_scores = np.asarray(real_scores, dtype=np.float64)
    fake_scores = np.asarray(fake_scores, dtype=np.float64)
    if not real_scores.size or not fake_scores.size:
        return {"auroc": np.nan, "accuracy": np.nan, "bce": np.nan}
    correct = np.concatenate((real_scores > 0.0, fake_scores <= 0.0))
    bce = np.mean(
        np.concatenate((np.logaddexp(0.0, -real_scores), np.logaddexp(0.0, fake_scores)))
    )
    return {
        "auroc": binary_auroc(real_scores, fake_scores),
        "accuracy": float(np.mean(correct)),
        "bce": float(bce),
    }


def select_case_study_indices(scores, n_samples, quantiles=None):
    """Choose distinct examples spanning the empirical fake-logit distribution."""
    scores = np.asarray(scores, dtype=np.float64)
    count = min(int(n_samples), len(scores))
    if count <= 0:
        return []
    if quantiles is None:
        quantiles = np.linspace(0.02, 0.98, count)
    else:
        quantiles = np.asarray(quantiles, dtype=np.float64)
        if len(quantiles) != count:
            quantiles = np.linspace(float(quantiles.min()), float(quantiles.max()), count)
    targets = np.quantile(scores, quantiles)
    available = np.ones(len(scores), dtype=bool)
    selected = []
    for quantile, target in zip(quantiles, targets):
        distances = np.abs(scores - target)
        distances[~available] = np.inf
        index = int(np.argmin(distances))
        available[index] = False
        selected.append((index, float(quantile)))
    return selected


def collect_case_study_samples(
    dataset, generator, fresh_discriminator, latitude_weights, target_rms,
    device, batch_size, num_workers, selected_indices, train_mean, train_std,
):
    """Re-run only selected held-out fields to retain maps without caching all fields."""
    requested = {index: quantile for index, quantile in selected_indices}
    if not requested:
        return []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    samples = []
    offset = 0
    generator.eval()
    fresh_discriminator.eval()
    with torch.no_grad():
        for clean in loader:
            clean = clean.to(device)
            generated, residual = generator(clean, latitude_weights, target_rms)
            batch_size_actual = clean.shape[0]
            for local_index, global_index in enumerate(range(offset, offset + batch_size_actual)):
                if global_index not in requested:
                    continue
                generated_logit, generated_map = fresh_discriminator.forward_with_logit_map(
                    generated[local_index:local_index + 1]
                )
                _, clean_map = fresh_discriminator.forward_with_logit_map(
                    clean[local_index:local_index + 1]
                )
                samples.append(
                    {
                        "index": global_index,
                        "logit_quantile": requested[global_index],
                        "fresh_fake_logit": float(generated_logit.item()),
                        "time": dataset.time_value(global_index),
                        "clean": (clean[local_index, 0] * float(train_std) + float(train_mean)).cpu().numpy(),
                        "generated": (generated[local_index, 0] * float(train_std) + float(train_mean)).cpu().numpy(),
                        "residual": (residual[local_index, 0] * float(train_std)).cpu().numpy(),
                        "clean_logit_map": clean_map[0, 0].cpu().numpy(),
                        "generated_logit_map": generated_map[0, 0].cpu().numpy(),
                    }
                )
            offset += batch_size_actual
    return sorted(samples, key=lambda sample: sample["logit_quantile"])


def evaluate_generator(
    cfg,
    dataset,
    generator,
    joint_discriminator,
    fresh_discriminator,
    latitude_weights,
    full_anchor_matrix,
    train_mean,
    train_std,
    target_rms,
    device,
    output_dir,
):
    loader = DataLoader(
        dataset,
        batch_size=int(experiment_get(cfg, "evaluation_batch_size", 32)),
        shuffle=False,
        num_workers=int(experiment_get(cfg, "num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    test_values = dataset.dataset[dataset.variable]
    test_mean = float(test_values.mean())
    test_std = max(float(test_values.std()), 1e-8)
    n_samples = len(dataset)
    n_anchors = full_anchor_matrix.shape[0]
    spectra = {key: None for key in ("clean", "generated", "random")}
    clean_split_spectra = {"even": None, "odd": None}
    constraint_values = {
        kind: {name: [] for name in ("rms", "effective_area_fraction", "peak_ratio")}
        for kind in ("generated", "random")
    }
    scores = {
        name: {kind: [] for kind in ("clean", "generated", "random")}
        for name in ("joint", "fresh")
    }
    random_generator = torch.Generator(device=device).manual_seed(
        int(experiment_get(cfg, "seed", 0)) + int(round(target_rms * 100000))
    )
    scratch_dir = experiment_get(cfg, "scratch_dir", None)
    if scratch_dir is not None:
        Path(str(scratch_dir)).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="adversarial-corruption-eval-",
        dir=None if scratch_dir is None else str(scratch_dir),
    ) as temporary_dir:
        response_arrays = {
            key: np.memmap(
                Path(temporary_dir) / f"{key}.dat",
                mode="w+",
                dtype=np.float32,
                shape=(n_samples, n_anchors),
            )
            for key in ("clean", "generated", "random")
        }
        generator.eval()
        joint_discriminator.eval()
        fresh_discriminator.eval()
        offset = 0
        with torch.no_grad():
            for clean in tqdm(loader, desc=f"Evaluate RMS {target_rms:.2f}"):
                clean = clean.to(device)
                generated, residual = generator(clean, latitude_weights, target_rms)
                random_fake, random_residual = random_coarse_perturbation(
                    clean,
                    latitude_weights,
                    target_rms,
                    experiment_get(cfg, "generator_coarsening", 4),
                    random_generator,
                )
                batch_size = clean.shape[0]
                for kind, values in (("generated", residual), ("random", random_residual)):
                    rms, area, peak = residual_statistics(values, latitude_weights)
                    constraint_values[kind]["rms"].extend(rms.cpu().numpy().reshape(-1).tolist())
                    constraint_values[kind]["effective_area_fraction"].extend(
                        area.cpu().numpy().reshape(-1).tolist()
                    )
                    constraint_values[kind]["peak_ratio"].extend(
                        peak.cpu().numpy().reshape(-1).tolist()
                    )
                metric_fields = {}
                for name, values in (
                    ("clean", clean),
                    ("generated", generated),
                    ("random", random_fake),
                ):
                    physical = values * float(train_std) + float(train_mean)
                    metric_fields[name] = (physical - test_mean) / test_std
                    flat = metric_fields[name][:, 0].reshape(batch_size, -1)
                    response = torch.sparse.mm(full_anchor_matrix, flat.T).T
                    response_arrays[name][offset:offset + batch_size] = response.cpu().numpy()
                    contribution = mean_zonal_spectrum(metric_fields[name], latitude_weights)
                    spectra[name] = contribution if spectra[name] is None else spectra[name] + contribution
                    scores["joint"][name].extend(joint_discriminator(values).cpu().numpy().reshape(-1).tolist())
                    scores["fresh"][name].extend(fresh_discriminator(values).cpu().numpy().reshape(-1).tolist())

                clean_sample_spectra = zonal_spectrum_per_sample(
                    metric_fields["clean"], latitude_weights
                ).cpu().numpy()
                global_indices = np.arange(offset, offset + batch_size)
                for split_name, mask in (
                    ("even", global_indices % 2 == 0),
                    ("odd", global_indices % 2 == 1),
                ):
                    if np.any(mask):
                        contribution = np.sum(clean_sample_spectra[mask], axis=0)
                        previous = clean_split_spectra[split_name]
                        clean_split_spectra[split_name] = (
                            contribution if previous is None else previous + contribution
                        )

                offset += batch_size

        for values in response_arrays.values():
            values.flush()
        even = np.arange(0, n_samples, 2)
        odd = np.arange(1, n_samples, 2)
        scwd_order = float(experiment_get(cfg, "scwd_order", 2.0))
        n_quantiles = int(experiment_get(cfg, "scwd_quantiles", 200))
        metrics = {
            "target_rms": target_rms,
            "n_train": 0,
            "n_test": n_samples,
            "test_scwd_generated": canonical_scwd(response_arrays["generated"], response_arrays["clean"], scwd_order, n_quantiles),
            "test_scwd_random": canonical_scwd(response_arrays["random"], response_arrays["clean"], scwd_order, n_quantiles),
            "test_scwd_clean_null": (
                canonical_scwd(
                    response_arrays["clean"][even],
                    response_arrays["clean"][odd],
                    scwd_order,
                    n_quantiles,
                )
                if len(even) and len(odd)
                else np.nan
            ),
        }
        reference_spectrum = spectra["clean"] / n_samples
        generated_spectrum = spectra["generated"] / n_samples
        random_spectrum = spectra["random"] / n_samples
        metrics["test_spectrum_generated"] = zonal_energy_spectrum_log_l2(
            generated_spectrum, reference_spectrum, cfg
        )
        metrics["test_spectrum_random"] = zonal_energy_spectrum_log_l2(
            random_spectrum, reference_spectrum, cfg
        )
        metrics["test_spectrum_l2_generated"] = relative_spectrum_l2(
            generated_spectrum, reference_spectrum
        )
        metrics["test_spectrum_l2_random"] = relative_spectrum_l2(
            random_spectrum, reference_spectrum
        )
        if len(even) and len(odd):
            even_spectrum = clean_split_spectra["even"] / len(even)
            odd_spectrum = clean_split_spectra["odd"] / len(odd)
            metrics["test_spectrum_clean_null"] = zonal_energy_spectrum_log_l2(
                even_spectrum, odd_spectrum, cfg
            )
            metrics["test_spectrum_l2_clean_null"] = relative_spectrum_l2(
                even_spectrum, odd_spectrum
            )
        else:
            metrics["test_spectrum_clean_null"] = np.nan
            metrics["test_spectrum_l2_clean_null"] = np.nan
        for kind in ("generated", "random"):
            for name, values in constraint_values[kind].items():
                metrics[f"test_{kind}_{name}"] = float(np.mean(values)) if values else np.nan
        for discriminator_name in ("joint", "fresh"):
            real_scores = scores[discriminator_name]["clean"]
            for fake_name in ("generated", "random"):
                fake_scores = scores[discriminator_name][fake_name]
                classification = discriminator_summary(real_scores, fake_scores)
                for name, value in classification.items():
                    metrics[f"{discriminator_name}_{name}_{fake_name}"] = value
                metrics[f"{discriminator_name}_mean_real_logit"] = float(np.mean(real_scores))
                metrics[f"{discriminator_name}_mean_{fake_name}_logit"] = float(np.mean(fake_scores))

    selected_indices = select_case_study_indices(
        scores["fresh"]["generated"],
        experiment_get(cfg, "case_study_samples", 6),
        experiment_get(cfg, "case_study_logit_quantiles", None),
    )
    selected = collect_case_study_samples(
        dataset, generator, fresh_discriminator, latitude_weights, target_rms, device,
        int(experiment_get(cfg, "evaluation_batch_size", 32)),
        int(experiment_get(cfg, "num_workers", 0)), selected_indices, train_mean, train_std,
    )
    write_case_study(selected, dataset.latitudes, dataset.longitudes, output_dir, target_rms)
    return metrics


def upsample_logit_map(values, shape):
    tensor = torch.as_tensor(values)[None, None]
    return functional.interpolate(tensor, size=shape, mode="nearest")[0, 0].numpy()


def write_case_study(samples, latitudes, longitudes, output_dir, target_rms):
    if not samples:
        return
    field_values = np.stack([sample[key] for sample in samples for key in ("clean", "generated")])
    field_min, field_max = np.percentile(field_values, [1, 99])
    residual_limit = max(float(np.percentile(np.abs([sample["residual"] for sample in samples]), 99)), 1e-6)
    map_values = np.stack(
        [
            upsample_logit_map(sample[key], sample["clean"].shape)
            for sample in samples
            for key in ("clean_logit_map", "generated_logit_map")
        ]
    )
    map_limit = max(float(np.percentile(np.abs(map_values), 99)), 1e-6)
    figure, axes = plt.subplots(
        len(samples),
        5,
        figsize=(25, 4 * len(samples)),
        squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    for row, sample in enumerate(samples):
        clean_map = upsample_logit_map(sample["clean_logit_map"], sample["clean"].shape)
        generated_map = upsample_logit_map(sample["generated_logit_map"], sample["clean"].shape)
        panels = (
            ("ERA5", sample["clean"], "inferno", field_min, field_max),
            ("Perturbed ERA5", sample["generated"], "inferno", field_min, field_max),
            ("Residual", sample["residual"], "RdBu_r", -residual_limit, residual_limit),
            ("ERA5 logit map", clean_map, "RdBu_r", -map_limit, map_limit),
            ("Perturbed logit map", generated_map, "RdBu_r", -map_limit, map_limit),
        )
        for column, (title, values, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[row, column]
            image = axis.pcolormesh(longitudes, latitudes, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            axis.set_global()
            axis.coastlines(linewidth=0.5)
            axis.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            axis.set_title(
                f"{title}\n{str(sample['time'])[:19]} | "
                f"fake logit {sample['fresh_fake_logit']:.2f} "
                f"(q={sample['logit_quantile']:.2f})",
                fontsize=9,
            )
            figure.colorbar(image, ax=axis, shrink=0.75)
    figure.suptitle(f"Learned metric blind-spot examples, normalized temperature RMS={target_rms:.2f}")
    figure.tight_layout(rect=[0, 0, 1, 0.98])
    figure.savefig(output_dir / "case_studies.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    dataset = xr.Dataset(
        data_vars={
            "era5": (("sample", "latitude", "longitude"), np.stack([item["clean"] for item in samples])),
            "perturbed_era5": (("sample", "latitude", "longitude"), np.stack([item["generated"] for item in samples])),
            "residual": (("sample", "latitude", "longitude"), np.stack([item["residual"] for item in samples])),
        },
        coords={
            "sample": np.arange(len(samples)),
            "time": ("sample", np.asarray([item["time"] for item in samples])),
            "fresh_fake_logit": ("sample", np.asarray([item["fresh_fake_logit"] for item in samples])),
            "logit_quantile": ("sample", np.asarray([item["logit_quantile"] for item in samples])),
            "latitude": latitudes,
            "longitude": longitudes,
        },
        attrs={
            "target_normalized_rms": float(target_rms),
            "field_units": "native ERA5 units",
            "residual_units": "native ERA5 units",
            "selection": "samples spaced across fresh-discriminator fake-logit quantiles",
        },
    )
    dataset.to_netcdf(output_dir / "case_studies.nc")
    dataset.close()


def write_rows(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_frontier(rows, output_path):
    if not rows:
        return
    rms = np.asarray([row["target_rms"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(rms, [row["test_scwd_generated"] for row in rows], marker="o", label="learned")
    axes[0].plot(rms, [row["test_scwd_random"] for row in rows], marker="o", label="random coarse")
    axes[0].set_ylabel("SCWD")
    axes[1].plot(rms, [row["test_spectrum_generated"] for row in rows], marker="o", label="learned")
    axes[1].plot(rms, [row["test_spectrum_random"] for row in rows], marker="o", label="random coarse")
    axes[1].set_ylabel("Log-spectrum L2")
    axes[2].plot(rms, [row["fresh_auroc_generated"] for row in rows], marker="o", label="learned")
    axes[2].plot(rms, [row["fresh_auroc_random"] for row in rows], marker="o", label="random coarse")
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Fresh SqueezeNet AUROC")
    for axis in axes:
        axis.set_xlabel("Perturbation RMS (train standard deviations)")
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("Metric-blind perturbation frontier")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def rms_tag(value):
    return f"rms_{float(value):.2f}".replace(".", "p")


@hydra.main(version_base=None, config_path="../conf", config_name="adversarial_corruption_config")
def main(cfg: DictConfig):
    validate_no_train_test_overlap(cfg)
    seed = int(experiment_get(cfg, "seed", 0))
    seed_everything(seed)
    device = resolve_device(cfg)
    variable = str(experiment_get(cfg, "variable", "2m_temperature"))
    output_root = Path(str(experiment_get(cfg, "output_dir"))) / variable
    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_root / "resolved_config.yaml", resolve=True)

    era5 = safe_open_dataset(cfg.real_nc_file)
    train_mean, train_std = training_normalization(era5, variable, cfg.train_real_range)
    train_dataset = ERA5TemperatureDataset(
        era5,
        variable,
        cfg.train_real_range,
        train_mean,
        train_std,
        experiment_get(cfg, "max_train_samples", 0),
    )
    test_dataset = ERA5TemperatureDataset(
        era5,
        variable,
        cfg.test_real_ranges,
        train_mean,
        train_std,
        experiment_get(cfg, "max_test_samples", 0),
    )
    latitude_weights = cosine_latitude_weights(train_dataset.latitudes, device=device)
    anchor_banks, full_anchor_matrix = build_anchor_banks(
        cfg, train_dataset.latitudes, train_dataset.longitudes, device
    )

    summaries = []
    for run_index, target_rms in enumerate(experiment_get(cfg, "rms_values", [0.05, 0.1, 0.2])):
        target_rms = float(target_rms)
        seed_everything(seed + run_index)
        output_dir = output_root / rms_tag(target_rms)
        generator, joint_discriminator, thresholds, _ = train_joint_pair(
            cfg,
            train_dataset,
            latitude_weights,
            anchor_banks,
            target_rms,
            device,
            output_dir,
        )
        fresh_discriminator = train_fresh_discriminator(
            cfg,
            train_dataset,
            generator,
            latitude_weights,
            target_rms,
            device,
            output_dir,
        )
        summary = evaluate_generator(
            cfg,
            test_dataset,
            generator,
            joint_discriminator,
            fresh_discriminator,
            latitude_weights,
            full_anchor_matrix,
            train_mean,
            train_std,
            target_rms,
            device,
            output_dir,
        )
        summary.update(
            {
                "n_train": len(train_dataset),
                "train_mean": train_mean,
                "train_std": train_std,
                "train_null95_scwd": thresholds["scwd"],
                "train_null95_spectrum": thresholds["spectrum"],
            }
        )
        write_rows([summary], output_dir / "test_summary.csv")
        summaries.append(summary)

    write_rows(summaries, output_root / "summary.csv")
    plot_frontier(summaries, output_root / "frontier.png")
    era5.close()
    print(f"Saved adversarial-corruption experiment to: {output_root}")


if __name__ == "__main__":
    main()
