"""Distributional spatial-realism baselines for forecast/corruption experiments.

The discriminator plots score whether samples look real, not whether individual
forecasts match paired ERA5 fields. This script therefore compares candidate
sample distributions against an ERA5 reference distribution. It deliberately
does not use per-sample forecast/ERA5 pairs. The configured variables are
standardized with ERA5 moments and evaluated jointly; configure a single
variable when field-wise metrics are desired.

Metrics:
- `mean_bias`: surface-area-weighted candidate mean minus ERA5 mean.
- `std_ratio_error`: surface-area-weighted candidate standard deviation divided by
  ERA5 standard deviation, minus one.
- `crps_like_field_energy`: half-energy distance between complete multi-field
  states, using cosine-latitude-weighted mean absolute differences.
- `zonal_energy_spectrum_l2`: relative L2 distance between mean zonal spectra.
- `sliced_wasserstein`: sliced Wasserstein distance between unweighted flattened
  spatial field distributions.
- `sliced_wasserstein_lon_corrected`: same metric after applying the spherical
  surface-Jacobian correction for lat-lon cell areas.
- `global_mean_wasserstein`: Vissio et al. quadratic Wasserstein distance
  between distributions of cosine-area-weighted global means.
- `mmd_rbf`: maximum mean discrepancy between flattened spatial-field
  distributions, using a Gaussian RBF kernel with a median-distance bandwidth.
- `scwd`: spherical convolutional Wasserstein distance approximation. Scalar
  fields use the compact-Wendland quantile approximation from Garrett et al.
  (2024); multi-field states use random channel weights inside each spherical
  filter so every filter still maps a sample to one scalar response.
- `scwd_area_weighted`: same SCWD responses, with cosine-area weighting when
  combining regular latitude-longitude anchor costs.
"""

import csv
import tempfile
import zlib
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import hydra
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


# Make independently labelled curves distinguishable without relying on colour.
SERIES_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "h")


def series_marker(index):
    return SERIES_MARKERS[int(index) % len(SERIES_MARKERS)]

try:
    from .plot_bundles import save_figure_bundle
    from .corruptions import U10_CHANNEL, V10_CHANNEL
    from .train_discriminator import (
        WeatherDiscriminator,
        apply_configured_corruption,
        normalize_prediction_timedelta,
        safe_open_dataset,
        select_time_ranges,
    )
    from .monthly_split import (
        concatenate_forecasts,
        coverage_metadata,
        evenly_spaced_pairs,
        forecast_pairs,
        select_era5_split,
    )
except ImportError:
    from plot_bundles import save_figure_bundle
    from corruptions import U10_CHANNEL, V10_CHANNEL
    from train_discriminator import (
        WeatherDiscriminator,
        apply_configured_corruption,
        normalize_prediction_timedelta,
        safe_open_dataset,
        select_time_ranges,
    )
    from monthly_split import (
        concatenate_forecasts,
        coverage_metadata,
        evenly_spaced_pairs,
        forecast_pairs,
        select_era5_split,
    )


DEFAULT_METRICS = [
    "mean_bias",
    "std_ratio_error",
    "crps_like_field_energy",
    "zonal_energy_spectrum_l2",
    "zonal_energy_spectrum_log_l2",
    "sliced_wasserstein",
    "sliced_wasserstein_lon_corrected",
    "global_mean_wasserstein",
    "mmd_rbf",
    "scwd_area_weighted",
    "scwd",
]

ERA5_NULL_LABEL = "ERA5 second-half null"
PLOTTING_DISABLED_METRICS = {
    "crps_like_field_energy",
    "sliced_wasserstein",
    "sliced_wasserstein_lon_corrected",
    "zonal_energy_spectrum_l2",
}


def plotted_metric_names(metric_names):
    """Exclude evaluation-only metrics from all standard figures."""
    return [name for name in metric_names if name not in PLOTTING_DISABLED_METRICS]


AVAILABLE_METRICS = {
    "mean_bias",
    "mean_abs_diff",
    "std_ratio_error",
    "std_abs_diff",
    "crps_like_field_energy",
    "zonal_energy_spectrum_l2",
    "zonal_energy_spectrum_log_l2",
    "sliced_wasserstein",
    "sliced_wasserstein_lon_corrected",
    "global_mean_wasserstein",
    "sliced_cramer_wold",
    "sliced_cramer_wold_lon_corrected",
    "mmd_rbf",
    "scwd_area_weighted",
    "scwd",
}

_SCWD_WEIGHT_CACHE = {}
_SCWD_TORCH_WEIGHT_CACHE = {}
_SCWD_NUMPY_WEIGHT_CACHE = {}

STRUCTURED_NEAR_NULL_CORRUPTIONS = {
    "equatorial_checker_texture",
    "meridional_scanlines",
    "checkerboard_2px",
    "zonal_scanlines",
}
DATA_DEPENDENT_CORRUPTIONS = {"hemisphere_splice"}


def cfg_get(cfg, key, default):
    """Return a config value, treating explicit YAML null as missing."""
    value = cfg.get(key)
    return default if value is None else value


def baseline_get(cfg, key, default=None):
    """Return a baseline setting, treating explicit YAML null as missing."""
    baseline = cfg.get("baseline")
    if baseline is None:
        raise ValueError("Missing required baseline configuration section.")
    value = baseline.get(key)
    return default if value is None else value


def torch_metric_device(cfg):
    """Return the configured Torch metric device, or None for NumPy metrics."""
    if cfg is None:
        return None
    backend = str(baseline_get(cfg, "backend", "auto")).lower()
    if backend in {"numpy", "np", "cpu"}:
        return None
    if backend not in {"auto", "torch", "cuda", "gpu"}:
        raise ValueError("baseline.backend must be one of: auto, torch, cuda, gpu, numpy")
    if backend == "auto" and not torch.cuda.is_available():
        return None
    configured_device = baseline_get(cfg, "device", None)
    if configured_device is None:
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(configured_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return None
    return device


def torch_metric_dtype(cfg):
    """Return the floating dtype for Torch metric kernels."""
    dtype_name = str(baseline_get(cfg, "dtype", "float32")).lower()
    if dtype_name in {"float64", "double"}:
        return torch.float64
    if dtype_name in {"float32", "single"}:
        return torch.float32
    raise ValueError("baseline.dtype must be float32 or float64")


def torch_tensor(values, device, dtype):
    """Move a NumPy-like array to the metric device."""
    return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)


def variables_from_config(cfg):
    """Return fields configured for the distribution baseline."""
    variables = baseline_get(cfg, "variables", None)
    if not variables:
        raise ValueError("baseline.variables must contain at least one field.")
    return list(variables)


def metric_names_from_config(cfg):
    """Return requested distributional baseline metrics."""
    metric_names = list(baseline_get(cfg, "metrics", DEFAULT_METRICS))
    unknown = sorted(set(metric_names) - AVAILABLE_METRICS)
    if unknown:
        raise ValueError(f"Unknown baseline metric(s): {unknown}. Available metrics: {sorted(AVAILABLE_METRICS)}")
    return metric_names


def select_level(ds, level):
    """Select a pressure level when the dataset has one."""
    if level is None:
        return ds
    if "level" in ds.dims:
        return ds.sel(level=level)
    if "pressure_level" in ds.dims:
        return ds.sel(pressure_level=level)
    return ds


def lead_hours(ds):
    """Return forecast lead times as integer hours."""
    ds = normalize_prediction_timedelta(ds)
    if "prediction_timedelta" not in ds.coords:
        return np.array([0], dtype=int)
    return np.asarray(ds.prediction_timedelta.values).astype(int)


def sample_time_indices(ds, max_samples):
    """Return evenly spaced time indices for an xarray dataset."""
    n_time = ds.sizes.get("time", 0)
    if n_time == 0:
        return []
    if max_samples <= 0 or n_time <= max_samples:
        return list(range(n_time))
    return np.linspace(0, n_time - 1, max_samples, dtype=int).tolist()


def pairwise_sample_positions(n_samples, max_samples):
    """Return deterministic positions retained by quadratic metrics."""
    if n_samples <= 0:
        return []
    if max_samples <= 0 or n_samples <= max_samples:
        return list(range(n_samples))
    return np.linspace(0, n_samples - 1, max_samples, dtype=int).tolist()


def corruption_sample_seed(base_seed, corruption_type, sample_key):
    """Return a stable per-corruption, per-sample seed shared across severities."""
    corruption_hash = zlib.crc32(str(corruption_type).encode("utf-8"))
    return int((int(base_seed) + corruption_hash + int(sample_key)) % (2**31 - 1))


def latitude_weights(latitudes):
    """Return cosine-latitude weights normalized by their mean."""
    weights = np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64)))
    return weights / np.nanmean(weights)


def latitude_weighted_moment_totals(values, latitudes):
    """Return surface-area-weighted sum, squared sum, and finite weight mass."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2] != len(latitudes):
        raise ValueError("Moment values must end in latitude-longitude dimensions.")
    weights = np.maximum(np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64))), 0.0)
    shape = (1,) * (values.ndim - 2) + (len(weights), 1)
    weights = weights.reshape(shape)
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, 0.0)
    weighted = weights * finite
    return (
        float(np.sum(safe_values * weighted, dtype=np.float64)),
        float(np.sum(safe_values * safe_values * weighted, dtype=np.float64)),
        float(np.sum(weighted, dtype=np.float64)),
    )



def area_weighted_global_means(fields, latitudes):
    """Return one cosine-area-weighted global mean per sample and field."""
    values = np.asarray(fields, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("Global-mean fields must have shape sample x channel x latitude x longitude.")
    weights = np.maximum(np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64))), 0.0)
    denominator = float(np.sum(weights) * values.shape[-1])
    if denominator <= 0.0:
        raise ValueError("Latitude weights have zero total area.")
    return np.sum(values * weights[None, None, :, None], axis=(-2, -1)) / denominator


def structured_near_null_pattern(corruption_type, latitudes, n_longitudes):
    """Return a zero-mean, unit-area-RMS spatial carrier for baseline probes."""
    n_latitudes = len(latitudes)
    rows, columns = np.indices((n_latitudes, int(n_longitudes)))
    if corruption_type == "equatorial_checker_texture":
        pattern = np.where((rows + columns) % 2, -1.0, 1.0)
        pattern *= 0.15 + 0.85 * np.exp(
            -(np.asarray(latitudes, dtype=np.float64)[:, None] / 25.0) ** 8
        )
    elif corruption_type == "meridional_scanlines":
        pattern = np.where(columns % 2, -1.0, 1.0)
    elif corruption_type == "checkerboard_2px":
        pattern = np.where(((rows // 2) + (columns // 2)) % 2, -1.0, 1.0)
    elif corruption_type == "zonal_scanlines":
        pattern = np.where(rows % 2, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown structured near-null corruption: {corruption_type}.")
    weights = np.maximum(np.cos(np.deg2rad(np.asarray(latitudes))), 0.0)[:, None]
    denominator = float(np.sum(weights) * int(n_longitudes))
    pattern -= np.sum(pattern * weights) / denominator
    rms = float(np.sqrt(np.sum(pattern**2 * weights) / denominator))
    if rms <= 1e-12:
        raise ValueError(f"Structured corruption {corruption_type} has zero energy.")
    return (pattern / rms).astype(np.float32)


def deranged_sample_positions(size, seed):
    """Return a reproducible cyclic donor permutation without self-pairs."""
    size = int(size)
    if size < 2:
        raise ValueError("Hemisphere splice requires at least two samples.")
    rng = np.random.default_rng(int(seed))
    return np.roll(np.arange(size, dtype=int), int(rng.integers(1, size)))


def apply_special_baseline_corruption(
    standardized,
    corruption_type,
    severity,
    latitudes,
    cfg,
    donor=None,
    maximum_severity=None,
    random_seed=None,
):
    """Apply structured or data-dependent corruptions in standardized space."""
    standardized = np.asarray(standardized, dtype=np.float32)
    severity = float(severity)
    if severity <= 0.0:
        return standardized
    if corruption_type in STRUCTURED_NEAR_NULL_CORRUPTIONS:
        pattern = structured_near_null_pattern(
            corruption_type, latitudes, standardized.shape[-1]
        )
        return standardized + severity * pattern[None, :, :]
    if corruption_type == "hemisphere_splice":
        if donor is None:
            raise ValueError("Hemisphere splice requires a donor ERA5 sample.")
        maximum = float(
            corruption_max_severity("hemisphere_splice", cfg)
            if maximum_severity is None else maximum_severity
        )
        replace_probability = np.clip(severity / max(maximum, 1e-12), 0.0, 1.0)
        boundary = float(baseline_get(cfg, "hemisphere_splice_latitude", 0.0))
        south = np.asarray(latitudes, dtype=np.float64) < boundary
        result = standardized.copy()
        rng = np.random.default_rng(random_seed) if random_seed is not None else np.random
        # The severity controls whether a *sample's whole southern hemisphere*
        # is replaced.  It is deliberately not a per-pixel blend probability.
        if rng.random() < replace_probability:
            result[:, south, :] = np.asarray(donor)[:, south, :]
        return result.astype(np.float32)
    raise ValueError(f"Unknown special baseline corruption: {corruption_type}.")


def weighted_quadratic_wasserstein_1d(support_a, mass_a, support_b, mass_b):
    """Compute exact one-dimensional W2 between two weighted discrete measures."""
    support_a = np.asarray(support_a, dtype=np.float64)
    support_b = np.asarray(support_b, dtype=np.float64)
    mass_a = np.asarray(mass_a, dtype=np.float64)
    mass_b = np.asarray(mass_b, dtype=np.float64)
    keep_a = mass_a > 0.0
    keep_b = mass_b > 0.0
    if not np.any(keep_a) or not np.any(keep_b):
        return np.nan
    support_a, mass_a = support_a[keep_a], mass_a[keep_a]
    support_b, mass_b = support_b[keep_b], mass_b[keep_b]
    mass_a = mass_a / np.sum(mass_a)
    mass_b = mass_b / np.sum(mass_b)
    cdf_a = np.cumsum(mass_a)
    cdf_b = np.cumsum(mass_b)
    boundaries = np.unique(np.concatenate(([0.0], cdf_a, cdf_b, [1.0])))
    widths = np.diff(boundaries)
    midpoints = boundaries[:-1] + 0.5 * widths
    idx_a = np.minimum(np.searchsorted(cdf_a, midpoints, side="right"), support_a.size - 1)
    idx_b = np.minimum(np.searchsorted(cdf_b, midpoints, side="right"), support_b.size - 1)
    return float(np.sqrt(np.sum(widths * (support_a[idx_a] - support_b[idx_b]) ** 2)))


def vissio_global_mean_wasserstein(candidate_means, reference_means, n_bins=20):
    """Return Ulam-binned W2 of global-mean time distributions.

    Test ERA5 fixes the equal-width phase-space bins. Candidate values beyond
    that fitted support are assigned to the nearest edge bin. For multiple
    fields, return the mean of their marginal one-dimensional distances.
    """
    candidate = np.asarray(candidate_means, dtype=np.float64)
    reference = np.asarray(reference_means, dtype=np.float64)
    if candidate.ndim == 1:
        candidate = candidate[:, None]
    if reference.ndim == 1:
        reference = reference[:, None]
    if candidate.ndim != 2 or reference.ndim != 2:
        return np.nan
    n_fields = min(candidate.shape[1], reference.shape[1])
    n_bins = int(n_bins)
    if n_fields == 0 or n_bins < 2:
        return np.nan

    distances = []
    support = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
    for field_idx in range(n_fields):
        cand = candidate[:, field_idx]
        ref = reference[:, field_idx]
        cand = cand[np.isfinite(cand)]
        ref = ref[np.isfinite(ref)]
        if cand.size == 0 or ref.size == 0:
            continue
        lower = float(np.min(ref))
        upper = float(np.max(ref))
        scale = upper - lower
        if scale <= 1e-12:
            distances.append(0.0 if np.allclose(cand, lower) else 1.0)
            continue
        padding = max(scale * 1e-9, np.finfo(np.float64).eps)
        edges = np.linspace(lower - padding, upper + padding, n_bins + 1)
        cand_hist = np.histogram(np.clip(cand, edges[0], edges[-1]), bins=edges)[0]
        ref_hist = np.histogram(ref, bins=edges)[0]
        distances.append(
            weighted_quadratic_wasserstein_1d(support, cand_hist, support, ref_hist)
        )
    return float(np.mean(distances)) if distances else np.nan


def zonal_energy_spectrum(values, latitudes):
    """Latitude-weighted zonal power spectrum for one 2D field."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    n_lat = len(latitudes)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D latitude-longitude field, got shape={values.shape}")
    if values.shape[0] != n_lat and values.shape[1] == n_lat:
        values = values.T
    if values.shape[0] != n_lat:
        raise ValueError(
            "Latitude dimension does not match field shape for zonal spectrum: "
            f"field={values.shape}, n_lat={n_lat}"
        )
    fft_values = np.fft.rfft(values, axis=-1)
    power = np.abs(fft_values) ** 2
    return np.average(power, axis=0, weights=latitude_weights(latitudes))


def zonal_energy_spectrum_log_l2(candidate_spectrum, reference_spectrum, cfg):
    """Root mean squared difference between log-power spectra."""
    common_spectrum_len = min(candidate_spectrum.size, reference_spectrum.size)
    if not common_spectrum_len:
        return np.nan
    candidate_values = np.maximum(np.asarray(candidate_spectrum[:common_spectrum_len], dtype=np.float64), 0.0)
    reference_values = np.maximum(np.asarray(reference_spectrum[:common_spectrum_len], dtype=np.float64), 0.0)
    finite_reference = reference_values[np.isfinite(reference_values)]
    positive_reference = finite_reference[finite_reference > 0.0]
    reference_scale = float(np.max(positive_reference)) if positive_reference.size else 1.0
    eps_factor = float(baseline_get(cfg, "spectrum_log_eps_factor", 1e-12))
    eps = max(reference_scale * eps_factor, np.finfo(np.float64).tiny)
    diff = np.log(candidate_values + eps) - np.log(reference_values + eps)
    return float(np.sqrt(np.nanmean(diff * diff))) if diff.size else np.nan


def canonical_latlon(values, latitudes):
    """Return a 2D field in latitude-longitude order."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    n_lat = len(latitudes)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D latitude-longitude field, got shape={values.shape}")
    if values.shape[0] != n_lat and values.shape[1] == n_lat:
        return values.T
    if values.shape[0] != n_lat:
        raise ValueError(f"Could not identify latitude axis: field={values.shape}, n_lat={n_lat}")
    return values


def invalid_field_reason(values, *, zero_atol=1e-12, min_std=1e-12):
    """Return why a field should be skipped, or None if it looks usable."""
    raw_values = np.asarray(values, dtype=np.float64)
    finite = raw_values[np.isfinite(raw_values)]
    if finite.size == 0:
        return "all_nonfinite"
    if np.all(np.abs(finite) <= zero_atol):
        return "all_zero"
    if float(np.nanstd(finite)) <= min_std:
        return "near_constant"
    return None


def should_skip_field(cfg, values):
    """Return true when invalid-field filtering should drop this field."""
    if not bool(baseline_get(cfg, "filter_invalid_fields", True)):
        return False
    zero_atol = float(baseline_get(cfg, "invalid_zero_atol", 1e-12))
    min_std = float(baseline_get(cfg, "invalid_min_std", 1e-12))
    return invalid_field_reason(values, zero_atol=zero_atol, min_std=min_std) is not None


def field_vector(values, latitudes, max_pixels):
    """Flatten one field with cosine-latitude area weighting."""
    values = canonical_latlon(values, latitudes)
    weighted = values * np.sqrt(latitude_weights(latitudes))[:, None]
    flat = weighted.ravel()
    if max_pixels <= 0 or flat.size <= max_pixels:
        return flat
    indices = np.linspace(0, flat.size - 1, max_pixels, dtype=int)
    return flat[indices]


def unweighted_field_vector(values, latitudes, max_pixels):
    """Flatten one field without surface-area correction."""
    values = canonical_latlon(values, latitudes)
    flat = values.ravel()
    if max_pixels <= 0 or flat.size <= max_pixels:
        return flat
    indices = np.linspace(0, flat.size - 1, max_pixels, dtype=int)
    return flat[indices]


def sliced_wasserstein_distance(candidate_vectors, reference_vectors, n_projections, seed, cfg=None):
    """Approximate Wasserstein distance with random 1D projections."""
    device = torch_metric_device(cfg)
    if device is not None:
        return sliced_wasserstein_distance_torch(candidate_vectors, reference_vectors, n_projections, seed, cfg, device)

    candidate = np.asarray(candidate_vectors, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    if candidate.size == 0 or reference.size == 0:
        return np.nan
    if candidate.ndim != 2 or reference.ndim != 2:
        return np.nan
    if candidate.shape[1] != reference.shape[1]:
        n_dim = min(candidate.shape[1], reference.shape[1])
        candidate = candidate[:, :n_dim]
        reference = reference[:, :n_dim]

    rng = np.random.default_rng(seed)
    distances = []
    n_quantiles = min(candidate.shape[0], reference.shape[0])
    for _ in range(n_projections):
        direction = rng.normal(size=candidate.shape[1])
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            continue
        direction /= norm
        candidate_proj = np.sort(candidate @ direction)
        reference_proj = np.sort(reference @ direction)
        candidate_q = np.linspace(0, candidate_proj.size - 1, n_quantiles)
        reference_q = np.linspace(0, reference_proj.size - 1, n_quantiles)
        candidate_values = np.interp(candidate_q, np.arange(candidate_proj.size), candidate_proj)
        reference_values = np.interp(reference_q, np.arange(reference_proj.size), reference_proj)
        distances.append(float(np.mean(np.abs(candidate_values - reference_values))))
    return float(np.mean(distances)) if distances else np.nan


def sliced_wasserstein_distance_torch(candidate_vectors, reference_vectors, n_projections, seed, cfg, device):
    """Approximate sliced Wasserstein with batched Torch projections."""
    candidate = np.asarray(candidate_vectors, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    if candidate.size == 0 or reference.size == 0:
        return np.nan
    if candidate.ndim != 2 or reference.ndim != 2:
        return np.nan
    n_dim = min(candidate.shape[1], reference.shape[1])
    if n_dim == 0:
        return np.nan
    candidate = candidate[:, :n_dim]
    reference = reference[:, :n_dim]

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(int(n_projections), n_dim))
    norms = np.linalg.norm(directions, axis=1)
    keep = norms > 1e-12
    if not np.any(keep):
        return np.nan
    directions = directions[keep] / norms[keep, None]

    dtype = torch_metric_dtype(cfg)
    candidate_t = torch_tensor(candidate, device, dtype)
    reference_t = torch_tensor(reference, device, dtype)
    directions_t = torch_tensor(directions.T, device, dtype)
    quantiles = torch.linspace(
        0.0,
        1.0,
        min(candidate_t.shape[0], reference_t.shape[0]),
        dtype=dtype,
        device=device,
    )

    candidate_proj = candidate_t @ directions_t
    reference_proj = reference_t @ directions_t
    candidate_q = torch.quantile(candidate_proj, quantiles, dim=0)
    reference_q = torch.quantile(reference_proj, quantiles, dim=0)
    return float(torch.mean(torch.abs(candidate_q - reference_q)).detach().cpu())


def sliced_wasserstein_from_projections(candidate_projections, reference_projections):
    """Compute sliced Wasserstein from compact, precomputed field projections."""
    candidate = np.asarray(candidate_projections, dtype=np.float64)
    reference = np.asarray(reference_projections, dtype=np.float64)
    if candidate.size == 0 or reference.size == 0:
        return np.nan
    if candidate.ndim != 2 or reference.ndim != 2:
        return np.nan
    n_projections = min(candidate.shape[1], reference.shape[1])
    n_quantiles = min(candidate.shape[0], reference.shape[0])
    if n_projections == 0 or n_quantiles == 0:
        return np.nan
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    candidate_q = np.quantile(candidate[:, :n_projections], quantiles, axis=0)
    reference_q = np.quantile(reference[:, :n_projections], quantiles, axis=0)
    return float(np.mean(np.abs(candidate_q - reference_q)))


def aligned_vector_sets(candidate_vectors, reference_vectors):
    """Return candidate/reference vector matrices with matching dimensions."""
    candidate = np.asarray(candidate_vectors, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    if candidate.size == 0 or reference.size == 0:
        return None, None
    if candidate.ndim != 2 or reference.ndim != 2:
        return None, None
    n_dim = min(candidate.shape[1], reference.shape[1])
    if n_dim == 0:
        return None, None
    return candidate[:, :n_dim], reference[:, :n_dim]


def standardize_for_mmd(candidate_vectors, reference_vectors, cfg):
    """Standardize MMD vectors using reference-distribution moments."""
    candidate, reference = aligned_vector_sets(candidate_vectors, reference_vectors)
    if candidate is None:
        return None, None
    if not bool(baseline_get(cfg, "mmd_standardize", True)):
        return candidate, reference

    mean = np.nanmean(reference, axis=0)
    std = np.nanstd(reference, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (candidate - mean) / std, (reference - mean) / std


def squared_euclidean_distances(left, right):
    """Compute pairwise squared Euclidean distances without materializing diffs."""
    left_norm = np.sum(left * left, axis=1)[:, None]
    right_norm = np.sum(right * right, axis=1)[None, :]
    distances = left_norm + right_norm - 2.0 * (left @ right.T)
    return np.maximum(distances, 0.0)


def mmd_rbf_bandwidth(reference, configured_bandwidth):
    """Fit an RBF bandwidth on the reference distribution alone."""
    if configured_bandwidth is not None and float(configured_bandwidth) > 0:
        return float(configured_bandwidth)
    distances = squared_euclidean_distances(reference, reference)
    positive = distances[distances > 1e-12]
    if positive.size == 0:
        return 1.0
    return float(np.sqrt(np.median(positive)))


def rbf_kernel_mean(left, right, bandwidth):
    """Mean Gaussian RBF kernel value between two vector samples."""
    distances = squared_euclidean_distances(left, right)
    scale = 2.0 * max(float(bandwidth), 1e-12) ** 2
    return float(np.mean(np.exp(-distances / scale)))


def mmd_rbf_distance(candidate_vectors, reference_vectors, cfg):
    """Biased RBF-MMD distance between candidate and reference field vectors."""
    candidate, reference = standardize_for_mmd(candidate_vectors, reference_vectors, cfg)
    if candidate is None:
        return np.nan
    device = torch_metric_device(cfg)
    if device is not None:
        return mmd_rbf_distance_torch(candidate, reference, cfg, device)
    bandwidth = mmd_rbf_bandwidth(reference, baseline_get(cfg, "mmd_bandwidth", None))
    k_xx = rbf_kernel_mean(candidate, candidate, bandwidth)
    k_yy = rbf_kernel_mean(reference, reference, bandwidth)
    k_xy = rbf_kernel_mean(candidate, reference, bandwidth)
    return float(np.sqrt(max(k_xx + k_yy - 2.0 * k_xy, 0.0)))


def squared_euclidean_distances_torch(left, right):
    """Compute pairwise squared Euclidean distances in Torch."""
    left_norm = torch.sum(left * left, dim=1, keepdim=True)
    right_norm = torch.sum(right * right, dim=1, keepdim=True).T
    return torch.clamp(left_norm + right_norm - 2.0 * (left @ right.T), min=0.0)


def mmd_rbf_bandwidth_torch(reference, configured_bandwidth):
    """Fit an RBF bandwidth on the reference distribution on the metric device."""
    if configured_bandwidth is not None and float(configured_bandwidth) > 0:
        return torch.as_tensor(float(configured_bandwidth), dtype=reference.dtype, device=reference.device)
    distances = squared_euclidean_distances_torch(reference, reference)
    positive = distances[distances > 1e-12]
    if positive.numel() == 0:
        return torch.as_tensor(1.0, dtype=candidate.dtype, device=candidate.device)
    return torch.sqrt(torch.quantile(positive, 0.5))


def rbf_kernel_mean_torch(left, right, bandwidth):
    """Mean Gaussian RBF kernel value between two vector samples in Torch."""
    distances = squared_euclidean_distances_torch(left, right)
    scale = 2.0 * torch.clamp(bandwidth, min=1e-12) ** 2
    return torch.mean(torch.exp(-distances / scale))


def mmd_rbf_distance_torch(candidate, reference, cfg, device):
    """Biased RBF-MMD distance with Torch pairwise distances."""
    dtype = torch_metric_dtype(cfg)
    candidate_t = torch_tensor(candidate, device, dtype)
    reference_t = torch_tensor(reference, device, dtype)
    bandwidth = mmd_rbf_bandwidth_torch(reference_t, baseline_get(cfg, "mmd_bandwidth", None))
    k_xx = rbf_kernel_mean_torch(candidate_t, candidate_t, bandwidth)
    k_yy = rbf_kernel_mean_torch(reference_t, reference_t, bandwidth)
    k_xy = rbf_kernel_mean_torch(candidate_t, reference_t, bandwidth)
    value = torch.sqrt(torch.clamp(k_xx + k_yy - 2.0 * k_xy, min=0.0))
    return float(value.detach().cpu())


def features_by_key(features):
    """Map sample keys to row indices for a finalized feature dictionary."""
    return {key: idx for idx, key in enumerate(features.get("sample_keys", []))}


def standardized_field(field, reference_features):
    """Standardize one field with the reference distribution for its variable."""
    std = reference_features["std"] if reference_features["std"] > 1e-12 else 1.0
    return (np.asarray(field, dtype=np.float64) - reference_features["mean"]) / std


def joint_features_from_variables(features_by_variable, reference_by_variable, variables, cfg):
    """Combine configured variables into one standardized multi-field feature set."""
    if not variables:
        raise ValueError("No variables configured for joint metric evaluation.")

    key_sets = []
    index_maps = {}
    for variable in variables:
        features = features_by_variable[variable]
        index_maps[variable] = features_by_key(features)
        key_sets.append(set(index_maps[variable]))

    common_keys = sorted(set.intersection(*key_sets)) if key_sets else []
    if not common_keys:
        return {
            "values": np.array([], dtype=np.float64),
            "sample_values": [],
            "spectrum": np.array([], dtype=np.float64),
            "spectra": np.empty((0, 0), dtype=np.float64),
            "fields": np.empty((0, 0, 0), dtype=np.float64),
            "channel_fields": np.empty((0, 0, 0, 0), dtype=np.float64),
            "vectors": np.empty((0, 0), dtype=np.float64),
            "unweighted_vectors": np.empty((0, 0), dtype=np.float64),
            "sample_keys": [],
            "n_valid_samples": 0,
            "latitudes": None,
            "longitudes": None,
            "mean_field": np.array([], dtype=np.float64),
            "channel_mean_fields": np.array([], dtype=np.float64),
            "mean": np.nan,
            "std": np.nan,
        }

    max_values = int(baseline_get(cfg, "value_samples", 20000))
    values_per_sample_variable = max(1, max_values // max(len(common_keys) * len(variables), 1))
    per_variable_pixels = max(1, int(baseline_get(cfg, "swd_pixels", 4096)) // len(variables))

    values_parts = []
    sample_values = []
    spectra = []
    fields = []
    channel_fields = []
    vectors = []
    unweighted_vectors = []
    latitudes = None
    longitudes = None

    for key in common_keys:
        sample_value_parts = []
        sample_spectra = []
        sample_fields = []
        sample_vectors = []
        sample_unweighted_vectors = []
        for variable in variables:
            source = features_by_variable[variable]
            reference = reference_by_variable[variable]
            field = source["fields"][index_maps[variable][key]]
            standardized = standardized_field(field, reference)
            sample_value_parts.append(subsample_flat_values(standardized, values_per_sample_variable))
            sample_spectra.append(zonal_energy_spectrum(standardized, source["latitudes"]))
            sample_fields.append(standardized)
            sample_vectors.append(field_vector(standardized, source["latitudes"], per_variable_pixels))
            sample_unweighted_vectors.append(unweighted_field_vector(standardized, source["latitudes"], per_variable_pixels))
            if latitudes is None:
                latitudes = source["latitudes"]
            if longitudes is None:
                longitudes = source["longitudes"]

        values = np.concatenate(sample_value_parts)
        values_parts.append(values)
        sample_values.append(values)
        spectra.append(np.concatenate(sample_spectra))
        channel_stack = np.stack(sample_fields)
        channel_fields.append(channel_stack)
        fields.append(np.concatenate(sample_fields, axis=1))
        vectors.append(np.concatenate(sample_vectors))
        unweighted_vectors.append(np.concatenate(sample_unweighted_vectors))

    all_values = np.concatenate(values_parts) if values_parts else np.array([], dtype=np.float64)
    spectra = np.stack(spectra) if spectra else np.empty((0, 0), dtype=np.float64)
    fields = np.stack(fields) if fields else np.empty((0, 0, 0), dtype=np.float64)
    channel_fields = np.stack(channel_fields) if channel_fields else np.empty((0, 0, 0, 0), dtype=np.float64)
    vectors = np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float64)
    unweighted_vectors = np.stack(unweighted_vectors) if unweighted_vectors else np.empty((0, 0), dtype=np.float64)

    return {
        "values": all_values,
        "sample_values": sample_values,
        "spectrum": np.nanmean(spectra, axis=0) if spectra.size else np.array([], dtype=np.float64),
        "spectra": spectra,
        "fields": fields,
        "channel_fields": channel_fields,
        "vectors": vectors,
        "unweighted_vectors": unweighted_vectors,
        "sample_keys": common_keys,
        "n_valid_samples": int(fields.shape[0]) if fields.ndim == 3 else 0,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "mean_field": np.nanmean(fields, axis=0) if fields.size else np.array([], dtype=np.float64),
        "channel_mean_fields": np.nanmean(channel_fields, axis=0) if channel_fields.size else np.array([], dtype=np.float64),
        "mean": float(np.nanmean(all_values)) if all_values.size else np.nan,
        "std": float(np.nanstd(all_values)) if all_values.size else np.nan,
    }


def regular_sphere_centers(n_lat, n_lon):
    """Return regular latitude-longitude cell centers for SCWD slicing."""
    latitudes = np.linspace(-90.0 + 90.0 / n_lat, 90.0 - 90.0 / n_lat, n_lat)
    longitudes = (np.arange(n_lon, dtype=np.float64) + 0.5) * 360.0 / n_lon
    return latitudes, longitudes


def nearest_indices(values, targets):
    """Return nearest indices in a possibly unsorted 1D coordinate array."""
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    order = np.argsort(values)
    sorted_values = values[order]
    right = np.searchsorted(sorted_values, targets, side="left")
    right = np.clip(right, 0, len(sorted_values) - 1)
    left = np.clip(right - 1, 0, len(sorted_values) - 1)
    choose_right = np.abs(sorted_values[right] - targets) < np.abs(sorted_values[left] - targets)
    return order[np.where(choose_right, right, left)]


def nearest_longitude_indices(longitudes, targets):
    """Return nearest periodic longitude indices."""
    longitudes = np.mod(np.asarray(longitudes, dtype=np.float64), 360.0)
    targets = np.mod(np.asarray(targets, dtype=np.float64), 360.0)
    order = np.argsort(longitudes)
    sorted_lons = longitudes[order]
    right = np.searchsorted(sorted_lons, targets, side="left") % len(sorted_lons)
    left = (right - 1) % len(sorted_lons)
    right_distance = np.abs(((sorted_lons[right] - targets + 180.0) % 360.0) - 180.0)
    left_distance = np.abs(((sorted_lons[left] - targets + 180.0) % 360.0) - 180.0)
    return order[np.where(right_distance < left_distance, right, left)]


def wendland_kernel(chordal_distance, radius_chordal):
    """Compact Wendland kernel from the SCWD paper, scaled to the range radius."""
    scaled = chordal_distance / max(float(radius_chordal), 1e-12)
    values = np.zeros_like(scaled, dtype=np.float64)
    mask = scaled <= 1.0
    d = scaled[mask]
    values[mask] = ((1.0 - d) ** 6) * (35.0 * d**2 + 18.0 * d + 3.0) / 3.0
    return values


def scwd_weight_vectors(reference, cfg):
    """Build and cache sparse Wendland convolution weights for one grid."""
    n_lat = int(baseline_get(cfg, "scwd_anchor_lat_points", 60))
    n_lon = int(baseline_get(cfg, "scwd_anchor_lon_points", 120))
    domain_n_lat = int(baseline_get(cfg, "scwd_domain_lat_points", 361))
    domain_n_lon = int(baseline_get(cfg, "scwd_domain_lon_points", 720))
    radius_km = float(baseline_get(cfg, "scwd_radius_km", 1000.0))
    latitudes = np.asarray(reference["latitudes"], dtype=np.float64)
    longitudes = np.asarray(reference["longitudes"], dtype=np.float64)
    cache_key = (
        tuple(np.round(latitudes, 8)),
        tuple(np.round(np.mod(longitudes, 360.0), 8)),
        n_lat,
        n_lon,
        domain_n_lat,
        domain_n_lon,
        radius_km,
    )
    if cache_key in _SCWD_WEIGHT_CACHE:
        return _SCWD_WEIGHT_CACHE[cache_key]

    earth_radius_km = 6371.0
    radius_radians = radius_km / earth_radius_km
    radius_degrees = np.rad2deg(radius_radians)
    radius_chordal = 2.0 * np.sin((radius_km / earth_radius_km) / 2.0)
    anchor_lats, anchor_lons = regular_sphere_centers(n_lat, n_lon)
    domain_lats = np.linspace(-90.0, 90.0, domain_n_lat)
    domain_lons = np.arange(domain_n_lon, dtype=np.float64) * 360.0 / domain_n_lon

    weights = []
    for lat0, lon0 in zip(np.deg2rad(np.repeat(anchor_lats, n_lon)), np.deg2rad(np.tile(anchor_lons, n_lat))):
        lat0_deg = np.rad2deg(lat0)
        lat_mask = (domain_lats >= lat0_deg - radius_degrees) & (domain_lats <= lat0_deg + radius_degrees)
        candidate_lats = domain_lats[lat_mask]
        if candidate_lats.size == 0:
            weights.append((np.array([], dtype=int), np.array([], dtype=np.float64)))
            continue

        lat_grid, lon_grid = np.meshgrid(candidate_lats, domain_lons, indexing="ij")
        flat_lat = np.deg2rad(lat_grid.ravel())
        flat_lon = np.deg2rad(lon_grid.ravel())
        cos_distance = (
            np.sin(lat0) * np.sin(flat_lat)
            + np.cos(lat0) * np.cos(flat_lat) * np.cos(flat_lon - lon0)
        )
        chordal_distance = np.sqrt(np.maximum(2.0 - 2.0 * np.clip(cos_distance, -1.0, 1.0), 0.0))
        kernel = wendland_kernel(chordal_distance, radius_chordal) * np.maximum(np.cos(flat_lat), 0.0)
        support = np.flatnonzero(kernel > 0)
        if support.size == 0:
            weights.append((support, kernel[support]))
            continue

        source_lat_idx = nearest_indices(latitudes, np.rad2deg(flat_lat[support]))
        source_lon_idx = nearest_longitude_indices(longitudes, np.rad2deg(flat_lon[support]))
        source_idx = source_lat_idx * len(longitudes) + source_lon_idx
        unique_source_idx, inverse = np.unique(source_idx, return_inverse=True)
        source_weights = np.bincount(inverse, weights=kernel[support], minlength=len(unique_source_idx))
        weights.append((unique_source_idx, source_weights / np.sum(source_weights)))

    _SCWD_WEIGHT_CACHE[cache_key] = weights
    return weights


def scwd_sparse_weight_matrix_torch(weights, n_pixels, cfg, device):
    """Return a cached sparse anchor-by-pixel SCWD weight matrix."""
    dtype = torch_metric_dtype(cfg)
    cache_key = (id(weights), int(n_pixels), str(device), str(dtype))
    if cache_key in _SCWD_TORCH_WEIGHT_CACHE:
        return _SCWD_TORCH_WEIGHT_CACHE[cache_key]

    rows = []
    cols = []
    values = []
    row_idx = 0
    for support, support_weights in weights:
        if support.size == 0:
            continue
        rows.append(np.full(support.size, row_idx, dtype=np.int64))
        cols.append(np.asarray(support, dtype=np.int64))
        values.append(np.asarray(support_weights, dtype=np.float64))
        row_idx += 1

    if not rows:
        matrix = None
    else:
        row_values = np.concatenate(rows)
        col_values = np.concatenate(cols)
        weight_values = np.concatenate(values)
        indices = torch.as_tensor(np.stack([row_values, col_values]), dtype=torch.long, device=device)
        data = torch.as_tensor(weight_values, dtype=dtype, device=device)
        matrix = torch.sparse_coo_tensor(indices, data, size=(row_idx, int(n_pixels)), device=device).coalesce()

    _SCWD_TORCH_WEIGHT_CACHE[cache_key] = matrix
    return matrix


def scwd_sparse_weight_matrix_numpy(weights, n_pixels):
    """Return a cached CSR anchor-by-pixel SCWD weight matrix."""
    cache_key = (id(weights), int(n_pixels))
    if cache_key in _SCWD_NUMPY_WEIGHT_CACHE:
        return _SCWD_NUMPY_WEIGHT_CACHE[cache_key]

    rows = []
    cols = []
    values = []
    for row_idx, (support, support_weights) in enumerate(weights):
        if support.size == 0:
            continue
        rows.append(np.full(support.size, row_idx, dtype=np.int64))
        cols.append(np.asarray(support, dtype=np.int64))
        values.append(np.asarray(support_weights, dtype=np.float32))
    if not rows:
        matrix = None
    else:
        matrix = scipy.sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(weights), int(n_pixels)),
            dtype=np.float32,
        )
    _SCWD_NUMPY_WEIGHT_CACHE[cache_key] = matrix
    return matrix


def scwd_response_batch(fields, numpy_weights, torch_weights, cfg, device):
    """Project a BxCxHxW field batch onto every SCWD spatial anchor."""
    fields = np.asarray(fields, dtype=np.float32)
    if device is None:
        flat = fields.reshape(fields.shape[0] * fields.shape[1], -1)
        responses = flat @ numpy_weights.T
        return np.asarray(responses, dtype=np.float32).reshape(fields.shape[0], fields.shape[1], -1)

    dtype = torch_metric_dtype(cfg)
    fields_t = torch_tensor(fields, device, dtype).reshape(fields.shape[0], fields.shape[1], -1)
    by_channel = [
        torch.sparse.mm(torch_weights, fields_t[:, channel_idx, :].T).T
        for channel_idx in range(fields.shape[1])
    ]
    return torch.stack(by_channel, dim=1).detach().cpu().numpy().astype(np.float32)


def scwd_channel_weights(cfg, n_channels):
    """Return the deterministic channel projections used by multi-field SCWD."""
    if n_channels == 1:
        return np.ones((1, 1), dtype=np.float64)
    rng = np.random.default_rng(int(baseline_get(cfg, "swd_seed", 0)))
    weights = rng.normal(
        size=(max(1, int(baseline_get(cfg, "scwd_channel_projections", 16))), n_channels)
    )
    norms = np.linalg.norm(weights, axis=1)
    weights = weights[norms > 1e-12]
    norms = norms[norms > 1e-12]
    return weights / norms[:, None] if weights.size else np.empty((0, n_channels), dtype=np.float64)


def scwd_anchor_transport_costs(candidate, reference, cfg):
    """Return the exact per-anchor r-powered contributions to disk-backed SCWD."""
    candidate_response = np.asarray(candidate.get("scwd_responses"))
    reference_response = np.asarray(reference.get("scwd_responses"))
    if candidate_response.size == 0 or reference_response.size == 0:
        return np.array([], dtype=np.float64)
    if candidate_response.ndim != 3 or reference_response.ndim != 3:
        return np.array([], dtype=np.float64)
    n_channels = min(candidate_response.shape[1], reference_response.shape[1])
    n_anchors = min(candidate_response.shape[2], reference_response.shape[2])
    if n_channels == 0 or n_anchors == 0:
        return np.array([], dtype=np.float64)

    r = float(baseline_get(cfg, "scwd_order", 2.0))
    n_quantiles = int(baseline_get(cfg, "scwd_quantiles", 200))
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    anchor_chunk = max(1, int(baseline_get(cfg, "scwd_anchor_chunk_size", 128)))
    channel_weights = scwd_channel_weights(cfg, n_channels)
    if channel_weights.size == 0:
        return np.array([], dtype=np.float64)

    costs = np.zeros(n_anchors, dtype=np.float64)
    for channel_weight in channel_weights:
        for start in range(0, n_anchors, anchor_chunk):
            stop = min(start + anchor_chunk, n_anchors)
            candidate_chunk = np.einsum(
                "nca,c->na", candidate_response[:, :n_channels, start:stop], channel_weight, optimize=True
            )
            reference_chunk = np.einsum(
                "nca,c->na", reference_response[:, :n_channels, start:stop], channel_weight, optimize=True
            )
            candidate_q = np.quantile(candidate_chunk, quantiles, axis=0)
            reference_q = np.quantile(reference_chunk, quantiles, axis=0)
            costs[start:stop] += np.mean(np.abs(candidate_q - reference_q) ** r, axis=0)
    return costs / len(channel_weights)


def scwd_anchor_w1_distributions(candidate, reference, cfg, n_top):
    """Rank anchors by local W1 and retain response samples for the strongest ones."""
    candidate_response = np.asarray(candidate.get("scwd_responses"))
    reference_response = np.asarray(reference.get("scwd_responses"))
    if candidate_response.size == 0 or reference_response.size == 0:
        return np.array([], dtype=np.float64), []
    if candidate_response.ndim != 3 or reference_response.ndim != 3:
        return np.array([], dtype=np.float64), []
    n_channels = min(candidate_response.shape[1], reference_response.shape[1])
    n_anchors = min(candidate_response.shape[2], reference_response.shape[2])
    if n_channels == 0 or n_anchors == 0:
        return np.array([], dtype=np.float64), []

    n_quantiles = int(baseline_get(cfg, "scwd_quantiles", 200))
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    anchor_chunk = max(1, int(baseline_get(cfg, "scwd_anchor_chunk_size", 128)))
    channel_weights = scwd_channel_weights(cfg, n_channels)
    if channel_weights.size == 0:
        return np.array([], dtype=np.float64), []

    local_w1 = np.zeros(n_anchors, dtype=np.float64)
    for channel_weight in channel_weights:
        for start in range(0, n_anchors, anchor_chunk):
            stop = min(start + anchor_chunk, n_anchors)
            candidate_chunk = np.einsum(
                "nca,c->na", candidate_response[:, :n_channels, start:stop], channel_weight, optimize=True
            )
            reference_chunk = np.einsum(
                "nca,c->na", reference_response[:, :n_channels, start:stop], channel_weight, optimize=True
            )
            candidate_q = np.quantile(candidate_chunk, quantiles, axis=0)
            reference_q = np.quantile(reference_chunk, quantiles, axis=0)
            local_w1[start:stop] += np.mean(np.abs(candidate_q - reference_q), axis=0)
    local_w1 /= len(channel_weights)

    n_top = min(max(int(n_top), 0), n_anchors)
    top_indices = np.argsort(local_w1)[::-1][:n_top]
    distributions = []
    for anchor_idx in top_indices:
        candidate_parts = [
            np.einsum(
                "nc,c->n",
                candidate_response[:, :n_channels, anchor_idx],
                channel_weight,
                optimize=True,
            )
            for channel_weight in channel_weights
        ]
        reference_parts = [
            np.einsum(
                "nc,c->n",
                reference_response[:, :n_channels, anchor_idx],
                channel_weight,
                optimize=True,
            )
            for channel_weight in channel_weights
        ]
        distributions.append(
            {
                "anchor_index": int(anchor_idx),
                "w1": float(local_w1[anchor_idx]),
                "candidate": np.concatenate(candidate_parts).astype(np.float32),
                "reference": np.concatenate(reference_parts).astype(np.float32),
            }
        )
    return local_w1, distributions

def scwd_anchor_mean_response_difference(candidate, reference, cfg):
    """Return candidate-minus-reference mean SCWD filter responses per anchor."""
    candidate_response = np.asarray(candidate.get("scwd_responses"))
    reference_response = np.asarray(reference.get("scwd_responses"))
    if candidate_response.size == 0 or reference_response.size == 0:
        return np.array([], dtype=np.float64)
    if candidate_response.ndim != 3 or reference_response.ndim != 3:
        return np.array([], dtype=np.float64)
    n_channels = min(candidate_response.shape[1], reference_response.shape[1])
    n_anchors = min(candidate_response.shape[2], reference_response.shape[2])
    channel_weights = scwd_channel_weights(cfg, n_channels)
    if n_channels == 0 or n_anchors == 0 or channel_weights.size == 0:
        return np.array([], dtype=np.float64)
    candidate_mean = np.mean(candidate_response[:, :n_channels, :n_anchors], axis=0)
    reference_mean = np.mean(reference_response[:, :n_channels, :n_anchors], axis=0)
    projected_difference = np.einsum(
        "pc,ca->pa", channel_weights, candidate_mean - reference_mean, optimize=True
    )
    return np.mean(projected_difference, axis=0)


def scwd_anchor_area_weights(cfg):
    """Return normalized surface-area weights for the regular SCWD anchor grid."""
    n_lat = int(baseline_get(cfg, "scwd_anchor_lat_points", 60))
    n_lon = int(baseline_get(cfg, "scwd_anchor_lon_points", 120))
    anchor_latitudes, _ = regular_sphere_centers(n_lat, n_lon)
    weights = np.repeat(np.maximum(np.cos(np.deg2rad(anchor_latitudes)), 0.0), n_lon)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("SCWD anchor grid has zero total surface-area weight.")
    return weights / total


def scwd_area_weighted_from_responses(candidate, reference, cfg):
    """Aggregate SCWD anchor transport costs using spherical surface-area weights."""
    costs = scwd_anchor_transport_costs(candidate, reference, cfg)
    if costs.size == 0:
        return np.nan
    weights = scwd_anchor_area_weights(cfg)
    if costs.size != weights.size:
        raise ValueError(f"SCWD anchor-weight mismatch: got {costs.size} costs and {weights.size} weights.")
    order = float(baseline_get(cfg, "scwd_order", 2.0))
    return float(np.sum(weights * costs) ** (1.0 / order))


def scwd_from_responses(candidate, reference, cfg):
    """Compute SCWD from disk-backed sample-by-channel-by-anchor responses."""
    costs = scwd_anchor_transport_costs(candidate, reference, cfg)
    order = float(baseline_get(cfg, "scwd_order", 2.0))
    return float(np.mean(costs) ** (1.0 / order)) if costs.size else np.nan


def scwd_filter_responses_torch(fields, weight_matrix, cfg, device):
    """Apply all SCWD spatial filters to a batch of fields."""
    dtype = torch_metric_dtype(cfg)
    fields_t = torch_tensor(fields, device, dtype).reshape(fields.shape[0], -1)
    return torch.sparse.mm(weight_matrix, fields_t.T).T


def spherical_convolutional_wasserstein_torch(candidate, reference, cfg, device):
    """Torch SCWD implementation for scalar fields."""
    candidate_fields = np.asarray(candidate["fields"], dtype=np.float64)
    reference_fields = np.asarray(reference["fields"], dtype=np.float64)
    if candidate_fields.size == 0 or reference_fields.size == 0:
        return np.nan
    if candidate_fields.ndim != 3 or reference_fields.ndim != 3:
        return np.nan
    if candidate_fields.shape[1:] != reference_fields.shape[1:]:
        return np.nan

    weights = scwd_weight_vectors(reference, cfg)
    weight_matrix = scwd_sparse_weight_matrix_torch(weights, candidate_fields.shape[1] * candidate_fields.shape[2], cfg, device)
    if weight_matrix is None or weight_matrix.shape[0] == 0:
        return np.nan

    dtype = torch_metric_dtype(cfg)
    r = float(baseline_get(cfg, "scwd_order", 2.0))
    n_quantiles = int(baseline_get(cfg, "scwd_quantiles", 200))
    quantiles = torch.linspace(0.0, 1.0, n_quantiles, dtype=dtype, device=device)

    candidate_response = scwd_filter_responses_torch(candidate_fields, weight_matrix, cfg, device)
    reference_response = scwd_filter_responses_torch(reference_fields, weight_matrix, cfg, device)
    candidate_quantiles = torch.quantile(candidate_response, quantiles, dim=0)
    reference_quantiles = torch.quantile(reference_response, quantiles, dim=0)
    total = torch.sum(torch.abs(candidate_quantiles - reference_quantiles) ** r)
    denom = float(weight_matrix.shape[0] * n_quantiles)
    return float(((total / denom) ** (1.0 / r)).detach().cpu())


def joint_spherical_convolutional_wasserstein_torch(candidate, reference, cfg, device):
    """Torch SCWD implementation for projected multi-channel fields."""
    candidate_fields = np.asarray(candidate["channel_fields"], dtype=np.float64)
    reference_fields = np.asarray(reference["channel_fields"], dtype=np.float64)
    if candidate_fields.size == 0 or reference_fields.size == 0:
        return np.nan
    if candidate_fields.ndim != 4 or reference_fields.ndim != 4:
        return np.nan
    if candidate_fields.shape[1:] != reference_fields.shape[1:]:
        return np.nan

    weights = scwd_weight_vectors(reference, cfg)
    n_pixels = candidate_fields.shape[2] * candidate_fields.shape[3]
    weight_matrix = scwd_sparse_weight_matrix_torch(weights, n_pixels, cfg, device)
    if weight_matrix is None or weight_matrix.shape[0] == 0:
        return np.nan

    dtype = torch_metric_dtype(cfg)
    n_samples, n_channels = candidate_fields.shape[:2]
    candidate_flat = torch_tensor(candidate_fields, device, dtype).reshape(n_samples, n_channels, n_pixels)
    reference_flat = torch_tensor(reference_fields, device, dtype).reshape(reference_fields.shape[0], n_channels, n_pixels)

    candidate_by_channel = torch.stack(
        [torch.sparse.mm(weight_matrix, candidate_flat[:, channel_idx, :].T).T for channel_idx in range(n_channels)],
        dim=1,
    )
    reference_by_channel = torch.stack(
        [torch.sparse.mm(weight_matrix, reference_flat[:, channel_idx, :].T).T for channel_idx in range(n_channels)],
        dim=1,
    )

    n_channel_projections = int(baseline_get(cfg, "scwd_channel_projections", 16))
    seed = int(baseline_get(cfg, "swd_seed", 0))
    rng = np.random.default_rng(seed)
    channel_weights = rng.normal(size=(max(1, n_channel_projections), n_channels))
    norms = np.linalg.norm(channel_weights, axis=1)
    keep = norms > 1e-12
    if not np.any(keep):
        return np.nan
    channel_weights = channel_weights[keep] / norms[keep, None]
    channel_weights_t = torch_tensor(channel_weights, device, dtype)

    candidate_slices = torch.einsum("sca,pc->psa", candidate_by_channel, channel_weights_t)
    reference_slices = torch.einsum("sca,pc->psa", reference_by_channel, channel_weights_t)

    r = float(baseline_get(cfg, "scwd_order", 2.0))
    n_quantiles = int(baseline_get(cfg, "scwd_quantiles", 200))
    quantiles = torch.linspace(0.0, 1.0, n_quantiles, dtype=dtype, device=device)
    candidate_quantiles = torch.quantile(candidate_slices, quantiles, dim=1)
    reference_quantiles = torch.quantile(reference_slices, quantiles, dim=1)
    total = torch.sum(torch.abs(candidate_quantiles - reference_quantiles) ** r)
    denom = float(channel_weights_t.shape[0] * weight_matrix.shape[0] * n_quantiles)
    return float(((total / denom) ** (1.0 / r)).detach().cpu())


def spherical_convolutional_wasserstein(candidate, reference, cfg):
    """Approximate SCWD using the quantile algorithm from Garrett et al. (2024)."""
    if "scwd_responses" in candidate and "scwd_responses" in reference:
        return scwd_from_responses(candidate, reference, cfg)
    if np.asarray(candidate.get("channel_fields", [])).ndim == 4:
        candidate_channels = np.asarray(candidate["channel_fields"], dtype=np.float64)
        reference_channels = np.asarray(reference["channel_fields"], dtype=np.float64)
        if candidate_channels.shape[1] > 1:
            return joint_spherical_convolutional_wasserstein(candidate, reference, cfg)

    candidate_fields = np.asarray(candidate["fields"], dtype=np.float64)
    reference_fields = np.asarray(reference["fields"], dtype=np.float64)
    if candidate_fields.size == 0 or reference_fields.size == 0:
        return np.nan
    if candidate_fields.ndim != 3 or reference_fields.ndim != 3:
        return np.nan
    if candidate_fields.shape[1:] != reference_fields.shape[1:]:
        return np.nan

    latitudes = reference["latitudes"]
    longitudes = reference["longitudes"]
    if latitudes is None or longitudes is None:
        return np.nan

    weights = scwd_weight_vectors(reference, cfg)
    if not weights:
        return np.nan

    device = torch_metric_device(cfg)
    if device is not None:
        return spherical_convolutional_wasserstein_torch(candidate, reference, cfg, device)

    r = float(baseline_get(cfg, "scwd_order", 2.0))
    n_quantiles = int(baseline_get(cfg, "scwd_quantiles", 200))
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    candidate_flat = candidate_fields.reshape(candidate_fields.shape[0], -1)
    reference_flat = reference_fields.reshape(reference_fields.shape[0], -1)

    total = 0.0
    n_used = 0
    for support, support_weights in weights:
        if support.size == 0:
            continue
        candidate_slice = candidate_flat[:, support] @ support_weights
        reference_slice = reference_flat[:, support] @ support_weights
        candidate_quantiles = np.quantile(candidate_slice, quantiles)
        reference_quantiles = np.quantile(reference_slice, quantiles)
        total += float(np.sum(np.abs(candidate_quantiles - reference_quantiles) ** r))
        n_used += 1
    # Approximate the SCWD integrals with means over convolution centers and
    # quantile locations. Summing here would make the value depend on the chosen
    # numerical grids rather than on the field distributions.
    return float((total / (n_used * n_quantiles)) ** (1.0 / r)) if n_used else np.nan


def joint_spherical_convolutional_wasserstein(candidate, reference, cfg):
    """Multi-channel SCWD with random channel weights inside each filter."""
    candidate_fields = np.asarray(candidate["channel_fields"], dtype=np.float64)
    reference_fields = np.asarray(reference["channel_fields"], dtype=np.float64)
    if candidate_fields.size == 0 or reference_fields.size == 0:
        return np.nan
    if candidate_fields.ndim != 4 or reference_fields.ndim != 4:
        return np.nan
    if candidate_fields.shape[1:] != reference_fields.shape[1:]:
        return np.nan

    weights = scwd_weight_vectors(reference, cfg)
    if not weights:
        return np.nan

    device = torch_metric_device(cfg)
    if device is not None:
        return joint_spherical_convolutional_wasserstein_torch(candidate, reference, cfg, device)

    n_channel_projections = int(baseline_get(cfg, "scwd_channel_projections", 16))
    seed = int(baseline_get(cfg, "swd_seed", 0))
    rng = np.random.default_rng(seed)
    candidate_flat = candidate_fields.reshape(candidate_fields.shape[0], candidate_fields.shape[1], -1)
    reference_flat = reference_fields.reshape(reference_fields.shape[0], reference_fields.shape[1], -1)
    r = float(baseline_get(cfg, "scwd_order", 2.0))
    n_quantiles = int(baseline_get(cfg, "scwd_quantiles", 200))
    quantiles = np.linspace(0.0, 1.0, n_quantiles)

    total = 0.0
    n_used = 0
    for support, support_weights in weights:
        if support.size == 0:
            continue
        candidate_by_channel = np.stack(
            [candidate_flat[:, channel_idx, support] @ support_weights for channel_idx in range(candidate_flat.shape[1])],
            axis=1,
        )
        reference_by_channel = np.stack(
            [reference_flat[:, channel_idx, support] @ support_weights for channel_idx in range(reference_flat.shape[1])],
            axis=1,
        )
        for _ in range(max(1, n_channel_projections)):
            channel_weights = rng.normal(size=candidate_by_channel.shape[1])
            norm = np.linalg.norm(channel_weights)
            if norm <= 1e-12:
                continue
            channel_weights /= norm
            candidate_slice = candidate_by_channel @ channel_weights
            reference_slice = reference_by_channel @ channel_weights
            candidate_quantiles = np.quantile(candidate_slice, quantiles)
            reference_quantiles = np.quantile(reference_slice, quantiles)
            total += float(np.sum(np.abs(candidate_quantiles - reference_quantiles) ** r))
            n_used += 1
    return float((total / (n_used * n_quantiles)) ** (1.0 / r)) if n_used else np.nan


def gaussian_density_inner_mean_1d(left, right, bandwidth):
    """Mean Gaussian-density inner product between two 1D empirical samples."""
    diffs = left[:, None] - right[None, :]
    variance = float(bandwidth) ** 2
    normalizer = np.sqrt(4.0 * np.pi * variance)
    return float(np.mean(np.exp(-(diffs ** 2) / (4.0 * variance)) / normalizer))


def projection_bandwidth(left, right, configured_bandwidth):
    """Choose Cramer-Wold bandwidth, falling back to a median-distance heuristic."""
    if configured_bandwidth is not None and float(configured_bandwidth) > 0:
        return float(configured_bandwidth)
    values = np.sort(np.concatenate([left, right]))
    if values.size < 2:
        return 1.0
    diffs = np.abs(values[:, None] - values[None, :])
    positive = diffs[diffs > 1e-12]
    if positive.size == 0:
        scale = float(np.std(values))
        return scale if scale > 1e-12 else 1.0
    return float(np.median(positive))


def sliced_cramer_wold_distance(candidate_vectors, reference_vectors, n_projections, seed, bandwidth=None):
    """Approximate sliced Cramer-Wold distance with Gaussian-kernel projections."""
    candidate = np.asarray(candidate_vectors, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    if candidate.size == 0 or reference.size == 0:
        return np.nan
    if candidate.ndim != 2 or reference.ndim != 2:
        return np.nan
    if candidate.shape[1] != reference.shape[1]:
        n_dim = min(candidate.shape[1], reference.shape[1])
        candidate = candidate[:, :n_dim]
        reference = reference[:, :n_dim]

    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(n_projections):
        direction = rng.normal(size=candidate.shape[1])
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            continue
        direction /= norm
        candidate_proj = candidate @ direction
        reference_proj = reference @ direction
        bw = projection_bandwidth(candidate_proj, reference_proj, bandwidth)
        k_xx = gaussian_density_inner_mean_1d(candidate_proj, candidate_proj, bw)
        k_yy = gaussian_density_inner_mean_1d(reference_proj, reference_proj, bw)
        k_xy = gaussian_density_inner_mean_1d(candidate_proj, reference_proj, bw)
        distances.append(float(np.sqrt(max(k_xx + k_yy - 2.0 * k_xy, 0.0))))
    return float(np.mean(distances)) if distances else np.nan


def field_l1_pairwise_mean(left_fields, right_fields, latitudes, chunk_size):
    """Mean cosine-latitude-weighted L1 distance over all field pairs."""
    left = np.asarray(left_fields, dtype=np.float32)
    right = np.asarray(right_fields, dtype=np.float32)
    if left.size == 0 or right.size == 0:
        return np.nan
    if left.ndim != 3 or right.ndim != 3 or left.shape[1:] != right.shape[1:]:
        return np.nan

    weights_1d = latitude_weights(latitudes).astype(np.float32)
    if left.shape[1] == weights_1d.size:
        weights = weights_1d
    elif left.shape[1] % weights_1d.size == 0:
        weights = np.tile(weights_1d, left.shape[1] // weights_1d.size)
    else:
        return np.nan
    weights = weights[None, :, None]
    chunk_size = max(1, int(chunk_size))
    total = 0.0
    n_pairs = 0
    for start in range(0, left.shape[0], chunk_size):
        chunk = left[start:start + chunk_size]
        distances = np.mean(np.abs(chunk[:, None, :, :] - right[None, :, :, :]) * weights, axis=(2, 3))
        total += float(np.sum(distances))
        n_pairs += int(distances.size)
    return total / n_pairs if n_pairs else np.nan


def cached_reference_field_self_distance(reference, cfg):
    """Return and cache E[d(Y,Y_prime)] on the reference feature object."""
    fields = np.asarray(reference["fields"])
    latitudes = reference["latitudes"]
    if fields.size == 0 or latitudes is None:
        return np.nan
    chunk_size = int(baseline_get(cfg, "field_energy_chunk_size", 4))
    cache = reference.setdefault("_field_self_distance_cache", {})
    cache_key = (tuple(np.round(np.asarray(latitudes, dtype=np.float64), 8)), chunk_size)
    if cache_key not in cache:
        cache[cache_key] = field_l1_pairwise_mean(fields, fields, latitudes, chunk_size)
    return cache[cache_key]


def crps_like_field_energy(candidate, reference, cfg):
    """Half-energy distance with a weighted L1 distance between complete fields."""
    fields = np.asarray(candidate["fields"])
    reference_fields = np.asarray(reference["fields"])
    latitudes = reference["latitudes"]
    if fields.size == 0 or reference_fields.size == 0 or latitudes is None:
        return np.nan

    chunk_size = int(baseline_get(cfg, "field_energy_chunk_size", 4))
    cross = field_l1_pairwise_mean(fields, reference_fields, latitudes, chunk_size)
    candidate_self = field_l1_pairwise_mean(fields, fields, latitudes, chunk_size)
    reference_self = cached_reference_field_self_distance(reference, cfg)
    return float(max(cross - 0.5 * candidate_self - 0.5 * reference_self, 0.0))


def subsample_flat_values(values, max_values):
    """Take a deterministic subset from a flattened field."""
    flat = np.ravel(np.asarray(values, dtype=np.float64))
    flat = flat[np.isfinite(flat)]
    if max_values <= 0 or flat.size <= max_values:
        return flat
    indices = np.linspace(0, flat.size - 1, max_values, dtype=int)
    return flat[indices]


def empty_features(variables):
    """Create mutable feature accumulators for each variable."""
    return {
        variable: {
            "values": [],
            "spectra": [],
            "fields": [],
            "vectors": [],
            "unweighted_vectors": [],
            "sample_keys": [],
            "latitudes": None,
            "longitudes": None,
        }
        for variable in variables
    }


def finalize_features(features, max_values):
    """Concatenate sampled values and average spectra for each variable."""
    finalized = {}
    for variable, parts in features.items():
        values = np.concatenate(parts["values"]) if parts["values"] else np.array([], dtype=np.float64)
        if max_values > 0 and values.size > max_values:
            values = values[np.linspace(0, values.size - 1, max_values, dtype=int)]
        spectra = np.stack(parts["spectra"]) if parts["spectra"] else np.empty((0, 0), dtype=np.float64)
        fields = np.stack(parts["fields"]) if parts["fields"] else np.empty((0, 0, 0), dtype=np.float64)
        vectors = np.stack(parts["vectors"]) if parts["vectors"] else np.empty((0, 0), dtype=np.float64)
        unweighted_vectors = (
            np.stack(parts["unweighted_vectors"]) if parts["unweighted_vectors"] else np.empty((0, 0), dtype=np.float64)
        )
        finalized[variable] = {
            "values": values,
            "sample_values": parts["values"],
            "spectrum": np.nanmean(spectra, axis=0) if spectra.size else np.array([], dtype=np.float64),
            "spectra": spectra,
            "fields": fields,
            "vectors": vectors,
            "unweighted_vectors": unweighted_vectors,
            "sample_keys": parts["sample_keys"],
            "n_valid_samples": int(fields.shape[0]) if fields.ndim == 3 else 0,
            "latitudes": parts["latitudes"],
            "longitudes": parts["longitudes"],
            "mean_field": np.nanmean(fields, axis=0) if fields.size else np.array([], dtype=np.float64),
            "mean": float(np.nanmean(values)) if values.size else np.nan,
            "std": float(np.nanstd(values)) if values.size else np.nan,
        }
    return finalized


def normalized_tensor(ds_slice, variables, means, stds):
    """Convert one ERA5 slice to the normalized tensor used by training corruptions."""
    channels = []
    for variable in variables:
        raw = ds_slice[variable].transpose("latitude", "longitude").values.astype(np.float32)
        channels.append(np.nan_to_num((raw - means[variable]) / stds[variable], nan=0.0))
    return torch.tensor(np.stack(channels), dtype=torch.float32)


def unnormalize_channel(channel, variable, means, stds):
    """Convert a normalized tensor channel back to physical units."""
    return channel.detach().cpu().numpy().astype(np.float32) * stds[variable] + means[variable]


def distribution_features(
    cfg,
    ds,
    variables,
    time_indices,
    max_values,
    *,
    lead_idx=None,
    corruption_type=None,
    severity=0.0,
    means=None,
    stds=None,
):
    """Extract unpaired distribution features from a dataset subset."""
    features = empty_features(variables)
    values_per_field = max(1, max_values // max(len(time_indices), 1)) if max_values > 0 else 0
    vector_pixels = int(baseline_get(cfg, "swd_pixels", 4096))

    for time_idx in time_indices:
        ds_slice = ds.isel(time=time_idx)
        if lead_idx is not None and "prediction_timedelta" in ds_slice.dims:
            ds_slice = ds_slice.isel(prediction_timedelta=lead_idx)

        corrupted_channels = None
        if corruption_type is not None:
            sample = normalized_tensor(ds_slice, variables, means, stds)
            corrupted_channels = apply_configured_corruption(sample, corruption_type, float(severity))

        for channel_idx, variable in enumerate(variables):
            if corrupted_channels is None:
                values = ds_slice[variable].transpose("latitude", "longitude").values.astype(np.float64)
            else:
                values = unnormalize_channel(corrupted_channels[channel_idx], variable, means, stds)
            latitudes = ds_slice[variable].latitude.values
            longitudes = ds_slice[variable].longitude.values
            if should_skip_field(cfg, values):
                continue
            values = canonical_latlon(values, latitudes)
            features[variable]["values"].append(subsample_flat_values(values, values_per_field))
            features[variable]["spectra"].append(zonal_energy_spectrum(values, latitudes))
            features[variable]["fields"].append(values)
            features[variable]["vectors"].append(field_vector(values, latitudes, vector_pixels))
            features[variable]["unweighted_vectors"].append(unweighted_field_vector(values, latitudes, vector_pixels))
            features[variable]["sample_keys"].append(time_idx)
            if features[variable]["latitudes"] is None:
                features[variable]["latitudes"] = np.asarray(latitudes, dtype=np.float64)
            if features[variable]["longitudes"] is None:
                features[variable]["longitudes"] = np.asarray(longitudes, dtype=np.float64)
    return finalize_features(features, max_values)


def streaming_reference_stats(cfg, ds, variables, time_indices):
    """Compute surface-area-weighted per-variable reference moments in bounded memory."""
    chunk_size = max(1, int(baseline_get(cfg, "feature_chunk_size", 32)))
    totals = {variable: [0.0, 0.0, 0.0] for variable in variables}
    for start in tqdm(range(0, len(time_indices), chunk_size), desc="ERA5 reference moments"):
        chunk = ds.isel(time=time_indices[start:start + chunk_size])
        for variable in variables:
            values = np.asarray(
                chunk[variable].transpose("time", "latitude", "longitude").values, dtype=np.float64
            )
            total, total_sq, weight_mass = latitude_weighted_moment_totals(
                values, chunk[variable].latitude.values
            )
            totals[variable][0] += total
            totals[variable][1] += total_sq
            totals[variable][2] += weight_mass
    stats = {}
    for variable, (total, total_sq, weight_mass) in totals.items():
        if weight_mass <= 0.0:
            raise ValueError(f"No finite reference values for {variable}.")
        mean = total / weight_mass
        variance = max(total_sq / weight_mass - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        stats[variable] = {"mean": float(mean), "std": std if std > 1e-8 else 1.0}
    return stats


def global_moment_correction(source_stats, target_stats, variables):
    """Fit one per-variable global affine forecast-to-ERA5 correction."""
    corrections = {}
    for variable in variables:
        source_mean = float(source_stats[variable]["mean"])
        source_std = float(source_stats[variable]["std"])
        target_mean = float(target_stats[variable]["mean"])
        target_std = float(target_stats[variable]["std"])
        if not np.isfinite(source_std) or source_std <= 1e-8:
            raise ValueError(
                f"Cannot moment-match {variable}: forecast training standard deviation "
                f"is {source_std!r}."
            )
        corrections[variable] = {
            "forecast_train_mean": source_mean,
            "forecast_train_std": source_std,
            "era5_train_mean": target_mean,
            "era5_train_std": target_std,
            "scale": target_std / source_std,
            "shift": target_mean - source_mean * target_std / source_std,
        }
    return corrections


def apply_global_moment_correction(values, correction):
    """Apply a train-fitted global affine forecast calibration."""
    return (
        (np.asarray(values, dtype=np.float32) - float(correction["forecast_train_mean"]))
        * float(correction["scale"]) + float(correction["era5_train_mean"])
    ).astype(np.float32)


def mmd_global_moment_matching_settings(cfg):
    """Configuration for the isolated train-fitted MMD calibration experiment."""
    return baseline_get(cfg, "mmd_global_moment_matching", {}) or {}


def mmd_global_moment_matching_output_dir(cfg, variables):
    settings = mmd_global_moment_matching_settings(cfg)
    return baseline_output_dir(cfg, variables) / str(
        settings.get("output_subdir", "mmd_global_moment_matching")
    )


def mmd_safe_filename(value):
    return "".join(character if character.isalnum() else "_" for character in str(value))


def write_mmd_global_moment_matching(rows, output_root):
    path = output_root / "data" / "mmd_global_moment_matching.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.unlink(missing_ok=True)
        return path
    fields = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_mmd_global_moment_matching(output_root):
    path = output_root / "data" / "mmd_global_moment_matching.csv"
    if not path.is_file():
        return []
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("lead_hour", "n_train_pairs", "n_test_pairs", "pairwise_n_samples"):
            row[key] = int(row[key])
        for key, value in list(row.items()):
            if key in {"label", "variables"} or key in {"lead_hour", "n_train_pairs", "n_test_pairs", "pairwise_n_samples"}:
                continue
            row[key] = float(value)
    return rows


def plot_mmd_global_moment_matching(rows, output_root):
    """Compare raw and train-moment-matched test MMD by forecast lead."""
    if not rows:
        return []
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    labels = sorted({row["label"] for row in rows})
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(labels), 1)))
    for label_index, (color, label) in enumerate(zip(colors, labels)):
        series = sorted((row for row in rows if row["label"] == label), key=lambda row: row["lead_hour"])
        leads = [row["lead_hour"] for row in series]
        axis.plot(leads, [row["raw_mmd_rbf"] for row in series], marker=series_marker(2 * label_index), color=color,
                  linestyle="--", alpha=0.7, label=f"{label} raw")
        axis.plot(leads, [row["matched_mmd_rbf"] for row in series], marker=series_marker(2 * label_index + 1), color=color,
                  linestyle="-", linewidth=2.0, label=f"{label} moment-matched")
    axis.set(
        xlabel="Lead time (hours)", ylabel="RBF MMD",
        title="Test MMD after train-fitted global mean/variance matching",
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    path = plot_root / "lead_time_raw_vs_global_moment_matched_mmd.png"
    save_figure_bundle(figure, path, plot_type="mmd_global_moment_matching", payload={"rows": np.asarray([str(row) for row in rows])}, dpi=220)
    plt.close(figure)
    return [path]


def evaluate_mmd_global_moment_matching(cfg):
    """Evaluate raw and train-fitted globally moment-matched forecast MMD only."""
    variables = variables_from_config(cfg)
    settings = mmd_global_moment_matching_settings(cfg)
    if not bool(settings.get("enabled", False)):
        return []
    output_root = mmd_global_moment_matching_output_dir(cfg, variables)
    real_ds = select_level(safe_open_dataset(cfg.real_nc_file), cfg.get("level"))
    requested_leads = {int(value) for value in cfg.lead_times}
    max_samples = int(settings.get("eval_samples", baseline_get(cfg, "eval_samples", 0)))
    rows = []
    scratch_dir = baseline_get(cfg, "scratch_dir", None)
    with tempfile.TemporaryDirectory(
        prefix="mmd-global-moment-matching-",
        dir=None if scratch_dir is None else str(scratch_dir),
    ) as temporary_dir:
        for label, configured_paths in baseline_get(cfg, "forecast_files", {}).items():
            opened = [normalize_prediction_timedelta(safe_open_dataset(path)) for path in configured_paths]
            forecast_ds = select_level(concatenate_forecasts(opened), cfg.get("level"))
            missing = [variable for variable in variables if variable not in forecast_ds.data_vars]
            if missing:
                print(f"Skipping {label}: missing variables {missing}")
                forecast_ds.close()
                continue
            for lead_idx, lead_hour in enumerate(lead_hours(forecast_ds)):
                lead_hour = int(lead_hour)
                if lead_hour not in requested_leads:
                    continue
                train_pairs = evenly_spaced_pairs(
                    [pair for pair in forecast_pairs(forecast_ds, real_ds, cfg, "train", [lead_hour])
                     if pair.lead_index == lead_idx], max_samples,
                )
                test_pairs = evenly_spaced_pairs(
                    [pair for pair in forecast_pairs(forecast_ds, real_ds, cfg, "test", [lead_hour])
                     if pair.lead_index == lead_idx], max_samples,
                )
                if not train_pairs or not test_pairs:
                    print(f"Skipping {label} +{lead_hour}h: missing paired train or test samples.")
                    continue
                train_forecast = forecast_ds.isel(prediction_timedelta=lead_idx)
                train_forecast_stats = streaming_reference_stats(
                    cfg, train_forecast, variables, [pair.forecast_index for pair in train_pairs]
                )
                train_era5 = real_ds.isel(time=[pair.era5_index for pair in train_pairs])
                train_era5_stats = streaming_reference_stats(
                    cfg, train_era5, variables, list(range(len(train_pairs)))
                )
                correction = global_moment_correction(train_forecast_stats, train_era5_stats, variables)
                test_era5 = real_ds.isel(time=[pair.era5_index for pair in test_pairs])
                test_indices = list(range(len(test_pairs)))
                test_stats = streaming_reference_stats(cfg, test_era5, variables, test_indices)
                reference = streaming_joint_features(
                    cfg, test_era5, variables, test_stats, test_indices, ["mmd_rbf"],
                    Path(temporary_dir) / f"{mmd_safe_filename(label)}_{lead_hour}_reference.dat",
                    description=f"{label} +{lead_hour}h ERA5 test MMD reference",
                )
                forecast_indices = [pair.forecast_index for pair in test_pairs]
                raw = streaming_joint_features(
                    cfg, forecast_ds, variables, test_stats, forecast_indices, ["mmd_rbf"],
                    Path(temporary_dir) / f"{mmd_safe_filename(label)}_{lead_hour}_raw.dat",
                    lead_idx=lead_idx, description=f"{label} +{lead_hour}h raw forecast MMD",
                )
                matched = streaming_joint_features(
                    cfg, forecast_ds, variables, test_stats, forecast_indices, ["mmd_rbf"],
                    Path(temporary_dir) / f"{mmd_safe_filename(label)}_{lead_hour}_matched.dat",
                    lead_idx=lead_idx, global_moment_correction=correction,
                    description=f"{label} +{lead_hour}h moment-matched forecast MMD",
                )
                raw_mmd = mmd_rbf_distance(raw["vectors"], reference["vectors"], cfg)
                matched_mmd = mmd_rbf_distance(matched["vectors"], reference["vectors"], cfg)
                row = {
                    "label": label, "variables": ",".join(variables), "lead_hour": lead_hour,
                    "n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs),
                    "pairwise_n_samples": min(len(raw["vectors"]), len(reference["vectors"])),
                    "raw_mmd_rbf": raw_mmd, "matched_mmd_rbf": matched_mmd,
                }
                for variable in variables:
                    for key, value in correction[variable].items():
                        row[f"{variable}__{key}"] = value
                rows.append(row)
                close_feature_memmaps(reference)
                close_feature_memmaps(raw)
                close_feature_memmaps(matched)
            forecast_ds.close()
    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=output_root / "resolved_config.yaml", resolve=True)
    csv_path = write_mmd_global_moment_matching(rows, output_root)
    real_ds.close()
    return [output_root / "resolved_config.yaml", csv_path] if csv_path.is_file() else [output_root / "resolved_config.yaml"]


def projection_directions(cfg, n_dimensions):
    """Return the deterministic directions used by both sliced-Wasserstein variants."""
    rng = np.random.default_rng(int(baseline_get(cfg, "swd_seed", 0)))
    directions = rng.normal(size=(int(baseline_get(cfg, "swd_projections", 64)), n_dimensions))
    norms = np.linalg.norm(directions, axis=1)
    directions = directions[norms > 1e-12]
    norms = norms[norms > 1e-12]
    return directions / norms[:, None]


def close_feature_memmaps(features):
    """Flush and close any disk-backed arrays owned by a compact feature summary."""
    for key in ("scwd_responses",):
        values = features.get(key)
        if isinstance(values, np.memmap):
            values.flush()
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()


def streaming_joint_features(
    cfg,
    ds,
    variables,
    reference_stats,
    time_indices,
    metric_names,
    scwd_path,
    *,
    lead_idx=None,
    corruption_type=None,
    severity=0.0,
    global_moment_correction=None,
    description="distribution features",
):
    """Build compact all-time metric summaries in bounded memory."""
    if lead_idx is not None and "prediction_timedelta" in ds.dims:
        ds = ds.isel(prediction_timedelta=lead_idx)
    if not time_indices:
        raise ValueError(f"No timestamps available for {description}.")
    requested = set(metric_names)
    pairwise_cap = int(baseline_get(cfg, "pairwise_eval_samples", 256))
    pairwise_positions = set(pairwise_sample_positions(len(time_indices), pairwise_cap))
    chunk_size = max(1, int(baseline_get(cfg, "feature_chunk_size", 32)))
    latitudes = np.asarray(ds[variables[0]].latitude.values, dtype=np.float64)
    longitudes = np.asarray(ds[variables[0]].longitude.values, dtype=np.float64)
    per_variable_pixels = max(1, int(baseline_get(cfg, "swd_pixels", 4096)) // len(variables))
    need_swd = bool({"sliced_wasserstein", "sliced_wasserstein_lon_corrected"} & requested)
    need_global_mean_wd = "global_mean_wasserstein" in requested
    need_scwd = bool({"scwd", "scwd_area_weighted"} & requested)
    need_pair_fields = "crps_like_field_energy" in requested
    need_pair_vectors = "mmd_rbf" in requested
    directions = None
    swd_projections = None
    swd_lon_projections = None
    pair_fields = []
    pair_vectors = []
    global_means = []
    pair_count = 0
    spectrum_total = None
    value_total = 0.0
    value_total_sq = 0.0
    value_weight_mass = 0.0
    n_valid = 0

    scwd_responses = None
    numpy_weights = None
    torch_weights = None
    device = None
    if need_scwd:
        reference_grid = {"latitudes": latitudes, "longitudes": longitudes}
        weights = scwd_weight_vectors(reference_grid, cfg)
        n_pixels = len(latitudes) * len(longitudes)
        device = torch_metric_device(cfg)
        if device is None:
            numpy_weights = scwd_sparse_weight_matrix_numpy(weights, n_pixels)
            n_anchors = 0 if numpy_weights is None else numpy_weights.shape[0]
        else:
            torch_weights = scwd_sparse_weight_matrix_torch(weights, n_pixels, cfg, device)
            n_anchors = 0 if torch_weights is None else torch_weights.shape[0]
        if n_anchors == 0:
            raise ValueError("SCWD produced no spatial anchors for the configured grid.")
        scwd_responses = np.memmap(
            scwd_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(time_indices), len(variables), n_anchors),
        )

    base_seed = int(baseline_get(cfg, "corruption_seed", 0))
    special_corruption = (
        corruption_type in STRUCTURED_NEAR_NULL_CORRUPTIONS | DATA_DEPENDENT_CORRUPTIONS
    )
    donor_positions = None
    if corruption_type == "hemisphere_splice":
        donor_positions = deranged_sample_positions(len(time_indices), base_seed)
    for start in tqdm(range(0, len(time_indices), chunk_size), desc=description):
        chunk_indices = time_indices[start:start + chunk_size]
        chunk_ds = ds.isel(time=chunk_indices)
        raw_by_variable = {
            variable: np.asarray(
                chunk_ds[variable].transpose("time", "latitude", "longitude").values,
                dtype=np.float32,
            )
            for variable in variables
        }
        donor_raw_by_variable = None
        if donor_positions is not None:
            donor_time_indices = [
                time_indices[int(donor_positions[position])]
                for position in range(start, start + len(chunk_indices))
            ]
            donor_chunk = ds.isel(time=donor_time_indices)
            donor_raw_by_variable = {
                variable: np.asarray(
                    donor_chunk[variable].transpose("time", "latitude", "longitude").values,
                    dtype=np.float32,
                )
                for variable in variables
            }
        valid_fields = []
        valid_positions = []
        for local_idx, time_idx in enumerate(chunk_indices):
            raw_fields = [raw_by_variable[variable][local_idx] for variable in variables]
            if global_moment_correction is not None:
                raw_fields = [
                    apply_global_moment_correction(
                        values, global_moment_correction[variable]
                    )
                    for variable, values in zip(variables, raw_fields)
                ]
            if any(should_skip_field(cfg, values) for values in raw_fields):
                continue
            donor_fields = None
            if donor_raw_by_variable is not None:
                donor_fields = [
                    donor_raw_by_variable[variable][local_idx] for variable in variables
                ]
                if any(should_skip_field(cfg, values) for values in donor_fields):
                    continue
            standardized = np.stack(
                [
                    (canonical_latlon(values, latitudes) - reference_stats[variable]["mean"])
                    / reference_stats[variable]["std"]
                    for variable, values in zip(variables, raw_fields)
                ]
            ).astype(np.float32)
            if corruption_type is not None:
                if special_corruption:
                    donor = None
                    if donor_fields is not None:
                        donor = np.stack(
                            [
                                (canonical_latlon(values, latitudes) - reference_stats[variable]["mean"])
                                / reference_stats[variable]["std"]
                                for variable, values in zip(variables, donor_fields)
                            ]
                        ).astype(np.float32)
                    standardized = apply_special_baseline_corruption(
                        standardized, corruption_type, severity, latitudes, cfg, donor,
                        random_seed=corruption_sample_seed(base_seed, corruption_type, time_idx),
                    )
                else:
                    seed = corruption_sample_seed(base_seed, corruption_type, time_idx)
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(seed)
                        standardized = apply_configured_corruption(
                            torch.from_numpy(standardized), corruption_type, float(severity)
                        ).detach().cpu().numpy().astype(np.float32)
            valid_fields.append(standardized)
            valid_positions.append(start + local_idx)

        if not valid_fields:
            continue
        field_batch = np.stack(valid_fields)
        batch_size = field_batch.shape[0]

        if need_global_mean_wd:
            global_means.append(area_weighted_global_means(field_batch, latitudes))

        total, total_sq, weight_mass = latitude_weighted_moment_totals(field_batch, latitudes)
        value_total += total
        value_total_sq += total_sq
        value_weight_mass += weight_mass

        batch_spectra = []
        weighted_vectors = []
        unweighted_vectors = []
        for sample_fields in field_batch:
            sample_spectra = [zonal_energy_spectrum(field, latitudes) for field in sample_fields]
            batch_spectra.append(np.concatenate(sample_spectra))
            if need_swd or need_pair_vectors:
                weighted_vectors.append(
                    np.concatenate(
                        [field_vector(field, latitudes, per_variable_pixels) for field in sample_fields]
                    )
                )
            if need_swd:
                unweighted_vectors.append(
                    np.concatenate(
                        [unweighted_field_vector(field, latitudes, per_variable_pixels) for field in sample_fields]
                    )
                )
        batch_spectrum_total = np.sum(np.stack(batch_spectra), axis=0, dtype=np.float64)
        spectrum_total = (
            batch_spectrum_total if spectrum_total is None else spectrum_total + batch_spectrum_total
        )

        if need_swd:
            weighted_vectors_array = np.asarray(weighted_vectors, dtype=np.float32)
            unweighted_vectors_array = np.asarray(unweighted_vectors, dtype=np.float32)
            if directions is None:
                directions = projection_directions(cfg, weighted_vectors_array.shape[1])
                n_projection = directions.shape[0]
                swd_projections = np.empty((len(time_indices), n_projection), dtype=np.float32)
                swd_lon_projections = np.empty((len(time_indices), n_projection), dtype=np.float32)
            swd_projections[n_valid:n_valid + batch_size] = unweighted_vectors_array @ directions.T
            swd_lon_projections[n_valid:n_valid + batch_size] = weighted_vectors_array @ directions.T
        if need_scwd:
            scwd_responses[n_valid:n_valid + batch_size] = scwd_response_batch(
                field_batch, numpy_weights, torch_weights, cfg, device
            )

        weighted_vectors_array = (
            np.asarray(weighted_vectors, dtype=np.float32) if weighted_vectors else None
        )
        for batch_idx, original_position in enumerate(valid_positions):
            if original_position not in pairwise_positions:
                continue
            pair_count += 1
            if need_pair_fields:
                pair_fields.append(np.concatenate(field_batch[batch_idx], axis=0))
            if need_pair_vectors:
                pair_vectors.append(weighted_vectors_array[batch_idx])
        n_valid += batch_size

    if n_valid == 0:
        raise ValueError(f"No valid fields available for {description}.")
    if value_weight_mass <= 0.0:
        raise ValueError(f"No finite weighted values available for {description}.")
    mean = value_total / value_weight_mass
    variance = max(value_total_sq / value_weight_mass - mean * mean, 0.0)
    if scwd_responses is not None:
        scwd_responses.flush()

    return {
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "spectrum": spectrum_total / n_valid,
        "swd_projections": (
            swd_projections[:n_valid] if swd_projections is not None else np.empty((0, 0), dtype=np.float32)
        ),
        "swd_lon_projections": (
            swd_lon_projections[:n_valid]
            if swd_lon_projections is not None
            else np.empty((0, 0), dtype=np.float32)
        ),
        "scwd_responses": (
            scwd_responses[:n_valid]
            if scwd_responses is not None
            else np.empty((0, 0, 0), dtype=np.float32)
        ),
        "fields": (
            np.stack(pair_fields).astype(np.float32)
            if pair_fields
            else np.empty((0, 0, 0), dtype=np.float32)
        ),
        "vectors": (
            np.stack(pair_vectors).astype(np.float32)
            if pair_vectors
            else np.empty((0, 0), dtype=np.float32)
        ),
        "unweighted_vectors": np.empty((0, 0), dtype=np.float32),
        "global_means": (
            np.concatenate(global_means, axis=0).astype(np.float32)
            if global_means
            else np.empty((0, len(variables)), dtype=np.float32)
        ),
        "latitudes": latitudes,
        "longitudes": longitudes,
        "n_valid_samples": int(n_valid),
        "pairwise_n_samples": int(pair_count),
    }


def distribution_metric_values(candidate, reference, cfg, metric_names=None):
    """Compute scalar distributional metrics from candidate/reference features."""
    requested = set(metric_names or AVAILABLE_METRICS)
    n_projections = int(baseline_get(cfg, "swd_projections", 64))
    seed = int(baseline_get(cfg, "swd_seed", 0))
    cramer_wold_bandwidth = baseline_get(cfg, "cramer_wold_bandwidth", None)
    candidate_unweighted = candidate.get("unweighted_vectors", candidate["vectors"])
    reference_unweighted = reference.get("unweighted_vectors", reference["vectors"])

    metrics = {}
    if "mean_bias" in requested:
        metrics["mean_bias"] = candidate["mean"] - reference["mean"]
    if "mean_abs_diff" in requested:
        metrics["mean_abs_diff"] = abs(candidate["mean"] - reference["mean"])
    if "std_ratio_error" in requested:
        metrics["std_ratio_error"] = candidate["std"] / (reference["std"] + 1e-12) - 1.0
    if "std_abs_diff" in requested:
        metrics["std_abs_diff"] = abs(candidate["std"] - reference["std"])
    if "crps_like_field_energy" in requested:
        metrics["crps_like_field_energy"] = crps_like_field_energy(candidate, reference, cfg)
    if {"zonal_energy_spectrum_l2", "zonal_energy_spectrum_log_l2"} & requested:
        candidate_spectrum = candidate["spectrum"]
        reference_spectrum = reference["spectrum"]
        common_spectrum_len = min(candidate_spectrum.size, reference_spectrum.size)
        if "zonal_energy_spectrum_l2" in requested:
            metrics["zonal_energy_spectrum_l2"] = (
                float(
                    np.linalg.norm(
                        candidate_spectrum[:common_spectrum_len] - reference_spectrum[:common_spectrum_len]
                    )
                    / (np.linalg.norm(reference_spectrum[:common_spectrum_len]) + 1e-12)
                )
                if common_spectrum_len
                else np.nan
            )
        if "zonal_energy_spectrum_log_l2" in requested:
            metrics["zonal_energy_spectrum_log_l2"] = zonal_energy_spectrum_log_l2(
                candidate_spectrum, reference_spectrum, cfg
            )
    if "sliced_wasserstein" in requested:
        if "swd_projections" in candidate and "swd_projections" in reference:
            metrics["sliced_wasserstein"] = sliced_wasserstein_from_projections(
                candidate["swd_projections"], reference["swd_projections"]
            )
        else:
            metrics["sliced_wasserstein"] = sliced_wasserstein_distance(
                candidate_unweighted, reference_unweighted, n_projections, seed, cfg
            )
    if "sliced_wasserstein_lon_corrected" in requested:
        if "swd_lon_projections" in candidate and "swd_lon_projections" in reference:
            metrics["sliced_wasserstein_lon_corrected"] = sliced_wasserstein_from_projections(
                candidate["swd_lon_projections"], reference["swd_lon_projections"]
            )
        else:
            metrics["sliced_wasserstein_lon_corrected"] = sliced_wasserstein_distance(
                candidate["vectors"], reference["vectors"], n_projections, seed, cfg
            )
    if "global_mean_wasserstein" in requested:
        metrics["global_mean_wasserstein"] = vissio_global_mean_wasserstein(
            candidate.get("global_means", np.empty((0, 0))),
            reference.get("global_means", np.empty((0, 0))),
            baseline_get(cfg, "global_mean_wd_bins", 20),
        )
    if "sliced_cramer_wold" in requested:
        metrics["sliced_cramer_wold"] = sliced_cramer_wold_distance(
            candidate_unweighted, reference_unweighted, n_projections, seed, cramer_wold_bandwidth
        )
    if "sliced_cramer_wold_lon_corrected" in requested:
        metrics["sliced_cramer_wold_lon_corrected"] = sliced_cramer_wold_distance(
            candidate["vectors"], reference["vectors"], n_projections, seed, cramer_wold_bandwidth
        )
    if "mmd_rbf" in requested:
        metrics["mmd_rbf"] = mmd_rbf_distance(candidate["vectors"], reference["vectors"], cfg)
    if "scwd_area_weighted" in requested:
        metrics["scwd_area_weighted"] = scwd_area_weighted_from_responses(candidate, reference, cfg)
    if "scwd" in requested:
        metrics["scwd"] = spherical_convolutional_wasserstein(candidate, reference, cfg)
    return metrics


def selected_distribution_metric_values(candidate, reference, metric_names, cfg):
    """Compute selected full-distribution metrics."""
    metrics = distribution_metric_values(candidate, reference, cfg, metric_names)
    metrics["n_samples"] = int(candidate.get("n_valid_samples", 0))
    metrics["pairwise_n_samples"] = int(candidate.get("pairwise_n_samples", 0))
    return metrics


def variable_tag(variables):
    """Return a collision-safe output tag for the configured fields."""
    return "__".join(str(variable).replace(" ", "_") for variable in variables)


def joint_variable_name(variables):
    """Return the output variable label for the configured joint sample."""
    return "all_fields" if len(variables) > 1 else variables[0]


def reference_features_from_config(cfg, truth_ds, variables, metric_names, temporary_dir):
    """Fit metric parameters on test ERA5 and retain train ERA5 for a shift diagnostic."""
    max_samples = int(baseline_get(cfg, "eval_samples", cfg_get(cfg, "max_samples", 100)))
    evaluation_range = baseline_get(cfg, "evaluation_time_range")
    evaluation_ds = select_level(select_time_ranges(truth_ds, evaluation_range), cfg.get("level"))
    evaluation_indices = sample_time_indices(evaluation_ds, max_samples)
    if not evaluation_indices:
        raise ValueError("No ERA5 samples in baseline.evaluation_time_range.")
    reference_stats = streaming_reference_stats(cfg, evaluation_ds, variables, evaluation_indices)
    evaluation_features = streaming_joint_features(
        cfg,
        evaluation_ds,
        variables,
        reference_stats,
        evaluation_indices,
        metric_names,
        Path(temporary_dir) / "evaluation-reference-scwd.dat",
        description="ERA5 test-reference features",
    )

    train_ranges = baseline_get(cfg, "reference_real_ranges", cfg_get(cfg, "train_real_range", None))
    train_ds = select_level(select_time_ranges(truth_ds, train_ranges), cfg.get("level"))
    train_indices = sample_time_indices(train_ds, max_samples)
    if not train_indices:
        raise ValueError("No ERA5 samples in baseline.reference_real_ranges.")
    train_features = streaming_joint_features(
        cfg,
        train_ds,
        variables,
        reference_stats,
        train_indices,
        metric_names,
        Path(temporary_dir) / "train-shift-reference-scwd.dat",
        description="ERA5 train-shift features",
    )
    return evaluation_features, train_features, reference_stats


def evaluate_era5_train_shift_metrics(evaluation_features, train_features, variables, metric_names, cfg):
    """Report the distinct ERA5 train-to-test distribution-shift diagnostic."""
    metrics = selected_distribution_metric_values(evaluation_features, train_features, metric_names, cfg)
    row = {
        "label": "ERA5 Test vs Train",
        "variable": joint_variable_name(variables),
        "lead_hour": 0,
        "n_samples": metrics["n_samples"],
        "pairwise_n_samples": metrics["pairwise_n_samples"],
    }
    row.update({metric_name: metrics[metric_name] for metric_name in metric_names})
    return row



def global_mean_wasserstein_diagnostic(candidate, reference, cfg, label, lead_hour, scalar_distance):
    """Retain global-mean samples needed to replot one forecast lead."""
    return {
        "label": str(label),
        "lead_hour": int(lead_hour),
        "candidate": np.asarray(candidate["global_means"], dtype=np.float32),
        "reference": np.asarray(reference["global_means"], dtype=np.float32),
        "distance": float(scalar_distance),
        "n_bins": int(baseline_get(cfg, "global_mean_wd_bins", 20)),
    }


def write_global_mean_wasserstein_diagnostics(diagnostics, variables, output_root):
    """Persist global-mean samples so plot-only runs do not reopen data."""
    output_path = output_root / "data" / "global_mean_wasserstein_distributions.nc"
    if not diagnostics:
        output_path.unlink(missing_ok=True)
        return
    n_fields = len(variables)
    candidate_size = max(item["candidate"].shape[0] for item in diagnostics)
    reference_size = max(item["reference"].shape[0] for item in diagnostics)
    candidate = np.full((len(diagnostics), n_fields, candidate_size), np.nan, dtype=np.float32)
    reference = np.full((len(diagnostics), n_fields, reference_size), np.nan, dtype=np.float32)
    candidate_count = np.zeros(len(diagnostics), dtype=np.int64)
    reference_count = np.zeros(len(diagnostics), dtype=np.int64)
    for comparison, item in enumerate(diagnostics):
        candidate_values = item["candidate"]
        reference_values = item["reference"]
        if candidate_values.shape[1] != n_fields or reference_values.shape[1] != n_fields:
            raise ValueError("Global-mean diagnostic field count does not match configured variables.")
        candidate_count[comparison] = candidate_values.shape[0]
        reference_count[comparison] = reference_values.shape[0]
        candidate[comparison, :, :candidate_values.shape[0]] = candidate_values.T
        reference[comparison, :, :reference_values.shape[0]] = reference_values.T
    dataset = xr.Dataset(
        data_vars={
            "candidate_global_mean": (("comparison", "field", "candidate_sample"), candidate),
            "reference_global_mean": (("comparison", "field", "reference_sample"), reference),
            "candidate_sample_count": ("comparison", candidate_count),
            "reference_sample_count": ("comparison", reference_count),
            "global_mean_wasserstein": (
                "comparison", np.asarray([item["distance"] for item in diagnostics], dtype=np.float64)
            ),
        },
        coords={
            "comparison": np.arange(len(diagnostics), dtype=np.int64),
            "label": ("comparison", [item["label"] for item in diagnostics]),
            "lead_hour": ("comparison", [item["lead_hour"] for item in diagnostics]),
            "field": list(variables),
        },
        attrs={
            "description": "Forecast and matched ERA5-test global-mean samples for Vissio global-mean W2 plots.",
            "n_bins": diagnostics[0]["n_bins"],
            "binning": "Equal-width bins fitted to each matched ERA5-test distribution; candidate values are clipped to its support.",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    dataset.close()
    print(f"Saved global-mean Wasserstein distributions to: {output_path}")


def read_global_mean_wasserstein_diagnostics(output_root):
    """Load persisted global-mean distributions for artifact-only plotting."""
    path = output_root / "data" / "global_mean_wasserstein_distributions.nc"
    if not path.exists():
        return []
    diagnostics = []
    with xr.open_dataset(path) as dataset:
        required = {"candidate_global_mean", "reference_global_mean", "candidate_sample_count", "reference_sample_count"}
        if not required.issubset(dataset.variables):
            raise ValueError(f"{path} predates artifact-only plotting; rerun baseline evaluation once.")
        fields = [str(value) for value in dataset.field.values]
        for comparison in range(dataset.sizes["comparison"]):
            n_candidate = int(dataset.candidate_sample_count.values[comparison])
            n_reference = int(dataset.reference_sample_count.values[comparison])
            diagnostics.append({
                "label": str(dataset.label.values[comparison]),
                "lead_hour": int(dataset.lead_hour.values[comparison]),
                "candidate": np.asarray(dataset.candidate_global_mean.values[comparison, :, :n_candidate]).T,
                "reference": np.asarray(dataset.reference_global_mean.values[comparison, :, :n_reference]).T,
                "distance": float(dataset.global_mean_wasserstein.values[comparison]),
                "n_bins": int(dataset.attrs.get("n_bins", 20)),
                "fields": fields,
            })
    return diagnostics


def plot_global_mean_wasserstein_distributions(diagnostics, output_root):
    """Overlay forecast and matched-ERA5 global-mean distributions by lead time."""
    if not diagnostics:
        return
    root = output_root / "plots" / "global_mean_wasserstein"
    root.mkdir(parents=True, exist_ok=True)
    for label in sorted({item["label"] for item in diagnostics}):
        series = sorted([item for item in diagnostics if item["label"] == label], key=lambda item: item["lead_hour"])
        fields = series[0]["fields"]
        n_panels = len(series) * len(fields)
        n_cols = min(2, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        figure, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.6 * n_rows), squeeze=False)
        flat_axes = axes.ravel()
        panel = 0
        for item in series:
            for field_index, field in enumerate(fields):
                axis = flat_axes[panel]
                candidate = item["candidate"][:, field_index]
                reference = item["reference"][:, field_index]
                candidate = candidate[np.isfinite(candidate)]
                reference = reference[np.isfinite(reference)]
                if candidate.size and reference.size:
                    lower, upper = np.min(reference), np.max(reference)
                    if upper > lower:
                        padding = max((upper - lower) * 1e-9, np.finfo(np.float64).eps)
                        edges = np.linspace(lower - padding, upper + padding, item["n_bins"] + 1)
                    else:
                        edges = np.histogram_bin_edges(np.concatenate([candidate, reference]), bins=max(2, item["n_bins"]))
                    axis.hist(reference, bins=edges, density=True, histtype="step", linewidth=1.8, color="black", label="ERA5 test")
                    axis.hist(np.clip(candidate, edges[0], edges[-1]), bins=edges, density=True, histtype="step", linewidth=1.8, color="tab:red", label=label)
                axis.set_title(f"+{item['lead_hour']} h | {field}\nGlobal-mean W2={item['distance']:.4g}", fontsize=10)
                axis.set_xlabel("Cosine-area-weighted global mean (standardized)")
                axis.set_ylabel("Density")
                axis.grid(True, alpha=0.25)
                panel += 1
        for axis in flat_axes[panel:]:
            axis.set_visible(False)
        flat_axes[0].legend(fontsize=9)
        figure.suptitle(f"Global-mean Wasserstein distributions: {label}", fontsize=14)
        figure.tight_layout(rect=[0, 0, 1, 0.94])
        output_path = root / f"{label.replace(' ', '_').replace('/', '_')}.png"
        payload = {"fields": np.asarray(fields)}
        for index, item in enumerate(series):
            payload[f"candidate_global_means_{index}"] = item["candidate"]
            payload[f"reference_global_means_{index}"] = item["reference"]
            payload[f"lead_hour_{index}"] = np.asarray(item["lead_hour"])
            payload[f"wasserstein_distance_{index}"] = np.asarray(item["distance"])
        save_figure_bundle(
            figure, output_path, plot_type="global_mean_wasserstein_distributions",
            payload=payload, dpi=220, bbox_inches="tight",
        )
        plt.close(figure)
        print(f"Saved global-mean Wasserstein distributions to: {output_path}")

def scwd_anchor_diagnostic(
    candidate, reference, cfg, label, lead_hour, scalar_scwd,
    comparison_kind="forecast", severity=None,
):
    """Build one labelled SCWD anchor-contribution diagnostic."""
    costs = scwd_anchor_transport_costs(candidate, reference, cfg)
    n_lat = int(baseline_get(cfg, "scwd_anchor_lat_points", 60))
    n_lon = int(baseline_get(cfg, "scwd_anchor_lon_points", 120))
    if costs.size != n_lat * n_lon:
        raise ValueError(
            f"SCWD anchor count mismatch: got {costs.size}, expected {n_lat} x {n_lon}."
        )
    r = float(baseline_get(cfg, "scwd_order", 2.0))
    anchor_lats, anchor_lons = regular_sphere_centers(n_lat, n_lon)
    reconstructed = float(np.mean(costs) ** (1.0 / r))
    if not np.isclose(reconstructed, scalar_scwd, rtol=1e-6, atol=1e-8):
        raise RuntimeError(
            f"SCWD anchor contributions do not reconstruct scalar value: {reconstructed} != {scalar_scwd}."
        )
    anchor_w1, top_w1_distributions = scwd_anchor_w1_distributions(
        candidate,
        reference,
        cfg,
        baseline_get(cfg, "scwd_top_w1_anchors", 6),
    )
    if anchor_w1.size != costs.size:
        raise RuntimeError("SCWD local-W1 anchor count does not match transport contributions.")
    mean_response_difference = scwd_anchor_mean_response_difference(candidate, reference, cfg)
    if mean_response_difference.size != costs.size:
        raise RuntimeError("SCWD mean-response anchor count does not match transport contributions.")
    for item in top_w1_distributions:
        anchor_lat_idx, anchor_lon_idx = divmod(item["anchor_index"], n_lon)
        item["latitude"] = float(anchor_lats[anchor_lat_idx])
        item["longitude"] = float(anchor_lons[anchor_lon_idx])
    return {
        "label": str(label),
        "comparison_kind": str(comparison_kind),
        "severity": np.nan if severity is None else float(severity),
        "lead_hour": int(lead_hour),
        "anchor_latitudes": anchor_lats,
        "anchor_longitudes": anchor_lons,
        "anchor_transport_cost": costs.reshape(n_lat, n_lon),
        "anchor_local_wasserstein": costs.reshape(n_lat, n_lon) ** (1.0 / r),
        "anchor_w1": anchor_w1.reshape(n_lat, n_lon),
        "anchor_mean_response_difference": mean_response_difference.reshape(n_lat, n_lon),
        "top_w1_distributions": top_w1_distributions,
        "scwd": float(scalar_scwd),
        "scwd_order": r,
    }


def write_scwd_anchor_diagnostics(diagnostics, output_root):
    """Write every value needed to recreate the SCWD diagnostic figures."""
    output_path = output_root / "data" / "scwd_anchor_contributions.nc"
    if not diagnostics:
        output_path.unlink(missing_ok=True)
        return
    first = diagnostics[0]
    top_count = max((len(item["top_w1_distributions"]) for item in diagnostics), default=0)
    response_count = max(
        (max(len(top["candidate"]), len(top["reference"]))
         for item in diagnostics for top in item["top_w1_distributions"]),
        default=0,
    )
    top_shape = (len(diagnostics), top_count)
    response_shape = (len(diagnostics), top_count, response_count)
    top_anchor_index = np.full(top_shape, -1, dtype=np.int64)
    top_w1 = np.full(top_shape, np.nan, dtype=np.float64)
    top_latitude = np.full(top_shape, np.nan, dtype=np.float64)
    top_longitude = np.full(top_shape, np.nan, dtype=np.float64)
    candidate_count = np.zeros(top_shape, dtype=np.int64)
    reference_count = np.zeros(top_shape, dtype=np.int64)
    candidate_response = np.full(response_shape, np.nan, dtype=np.float32)
    reference_response = np.full(response_shape, np.nan, dtype=np.float32)
    for comparison, item in enumerate(diagnostics):
        for rank, top in enumerate(item["top_w1_distributions"]):
            candidate = np.asarray(top["candidate"], dtype=np.float32)
            reference = np.asarray(top["reference"], dtype=np.float32)
            top_anchor_index[comparison, rank] = int(top["anchor_index"])
            top_w1[comparison, rank] = float(top["w1"])
            top_latitude[comparison, rank] = float(top["latitude"])
            top_longitude[comparison, rank] = float(top["longitude"])
            candidate_count[comparison, rank] = len(candidate)
            reference_count[comparison, rank] = len(reference)
            candidate_response[comparison, rank, :len(candidate)] = candidate
            reference_response[comparison, rank, :len(reference)] = reference

    dataset = xr.Dataset(
        data_vars={
            "anchor_transport_cost": (
                ("comparison", "anchor_latitude", "anchor_longitude"),
                np.stack([item["anchor_transport_cost"] for item in diagnostics]).astype(np.float64),
            ),
            "anchor_local_wasserstein": (
                ("comparison", "anchor_latitude", "anchor_longitude"),
                np.stack([item["anchor_local_wasserstein"] for item in diagnostics]).astype(np.float64),
            ),
            "anchor_w1": (
                ("comparison", "anchor_latitude", "anchor_longitude"),
                np.stack([item["anchor_w1"] for item in diagnostics]).astype(np.float64),
            ),
            "anchor_mean_response_difference": (
                ("comparison", "anchor_latitude", "anchor_longitude"),
                np.stack([item["anchor_mean_response_difference"] for item in diagnostics]).astype(np.float64),
            ),
            "scwd": ("comparison", np.asarray([item["scwd"] for item in diagnostics], dtype=np.float64)),
            "top_anchor_index": (("comparison", "top_rank"), top_anchor_index),
            "top_w1": (("comparison", "top_rank"), top_w1),
            "top_latitude": (("comparison", "top_rank"), top_latitude),
            "top_longitude": (("comparison", "top_rank"), top_longitude),
            "candidate_response_count": (("comparison", "top_rank"), candidate_count),
            "reference_response_count": (("comparison", "top_rank"), reference_count),
            "candidate_response": (("comparison", "top_rank", "response_sample"), candidate_response),
            "reference_response": (("comparison", "top_rank", "response_sample"), reference_response),
        },
        coords={
            "comparison": np.arange(len(diagnostics), dtype=np.int64),
            "label": ("comparison", [item["label"] for item in diagnostics]),
            "comparison_kind": ("comparison", [item.get("comparison_kind", "forecast") for item in diagnostics]),
            "severity": ("comparison", [item.get("severity", np.nan) for item in diagnostics]),
            "lead_hour": ("comparison", [item["lead_hour"] for item in diagnostics]),
            "anchor_latitude": first["anchor_latitudes"],
            "anchor_longitude": first["anchor_longitudes"],
        },
        attrs={
            "description": "SCWD per-anchor transport diagnostics for forecast leads and full-strength corruptions.",
            "scwd_order": first["scwd_order"],
            "reconstruction": "scwd = mean(anchor_transport_cost) ** (1 / scwd_order)",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    dataset.close()
    print(f"Saved SCWD anchor diagnostics to: {output_path}")


def read_scwd_anchor_diagnostics(output_root):
    """Load SCWD diagnostics without reopening forecast or ERA5 datasets."""
    path = output_root / "data" / "scwd_anchor_contributions.nc"
    if not path.exists():
        return []
    diagnostics = []
    with xr.open_dataset(path) as dataset:
        required = {"candidate_response", "reference_response", "top_anchor_index"}
        if not required.issubset(dataset.variables):
            raise ValueError(f"{path} predates artifact-only plotting; rerun baseline evaluation once.")
        for comparison in range(dataset.sizes["comparison"]):
            distributions = []
            for rank in range(dataset.sizes.get("top_rank", 0)):
                anchor_index = int(dataset.top_anchor_index.values[comparison, rank])
                if anchor_index < 0:
                    continue
                n_candidate = int(dataset.candidate_response_count.values[comparison, rank])
                n_reference = int(dataset.reference_response_count.values[comparison, rank])
                distributions.append({
                    "anchor_index": anchor_index,
                    "w1": float(dataset.top_w1.values[comparison, rank]),
                    "latitude": float(dataset.top_latitude.values[comparison, rank]),
                    "longitude": float(dataset.top_longitude.values[comparison, rank]),
                    "candidate": np.asarray(dataset.candidate_response.values[comparison, rank, :n_candidate]),
                    "reference": np.asarray(dataset.reference_response.values[comparison, rank, :n_reference]),
                })
            diagnostics.append({
                "label": str(dataset.label.values[comparison]),
                "comparison_kind": (str(dataset.comparison_kind.values[comparison])
                                    if "comparison_kind" in dataset.coords else "forecast"),
                "severity": (float(dataset.severity.values[comparison])
                             if "severity" in dataset.coords else np.nan),
                "lead_hour": int(dataset.lead_hour.values[comparison]),
                "anchor_latitudes": np.asarray(dataset.anchor_latitude.values),
                "anchor_longitudes": np.asarray(dataset.anchor_longitude.values),
                "anchor_transport_cost": np.asarray(dataset.anchor_transport_cost.values[comparison]),
                "anchor_local_wasserstein": np.asarray(dataset.anchor_local_wasserstein.values[comparison]),
                "anchor_w1": np.asarray(dataset.anchor_w1.values[comparison]),
                "anchor_mean_response_difference": np.asarray(dataset.anchor_mean_response_difference.values[comparison]),
                "top_w1_distributions": distributions,
                "scwd": float(dataset.scwd.values[comparison]),
                "scwd_order": float(dataset.attrs["scwd_order"]),
            })
    return diagnostics


def scwd_comparison_kind(item):
    return str(item.get("comparison_kind", "forecast"))


def scwd_comparison_detail(item):
    comparison_kind = scwd_comparison_kind(item)
    if comparison_kind == "corruption":
        return f"severity={float(item['severity']):.3g}"
    if comparison_kind == "null":
        return "ERA5 days 16–31 vs days 1–15"
    return f"+{int(item['lead_hour'])} h"


def scwd_plot_root(output_root, comparison_kind, top_w1=False):
    root = output_root / "plots" / "scwd"
    if comparison_kind == "corruption":
        root = root / "corruptions"
    elif comparison_kind == "null":
        root = root / "null"
    if top_w1:
        root = root / "top_w1_distributions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def scwd_output_stem(item):
    label = str(item["label"]).replace(" ", "_").replace("/", "_")
    if scwd_comparison_kind(item) == "corruption":
        return f"{label}_severity_{float(item['severity']):.3g}"
    return label


def plot_scwd_anchor_diagnostics(diagnostics, output_root):
    """Plot shared-scale SCWD local-Wasserstein maps in one figure per model."""
    if not diagnostics:
        return
    values = np.stack([item["anchor_local_wasserstein"] for item in diagnostics])
    vmax = max(float(np.nanpercentile(values, 99)), 1e-12)
    labels = sorted({(scwd_comparison_kind(item), item["label"]) for item in diagnostics})

    for comparison_kind, label in labels:
        series = sorted(
            [item for item in diagnostics
             if scwd_comparison_kind(item) == comparison_kind and item["label"] == label],
            key=lambda item: item["lead_hour"],
        )
        n_cols = min(2, len(series))
        n_rows = int(np.ceil(len(series) / n_cols))
        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.2 * n_cols, 3.6 * n_rows),
            squeeze=False,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        flat_axes = axes.ravel()
        image = None
        for axis, item in zip(flat_axes, series):
            image = axis.pcolormesh(
                item["anchor_longitudes"],
                item["anchor_latitudes"],
                item["anchor_local_wasserstein"],
                shading="auto",
                transform=ccrs.PlateCarree(),
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
            axis.set_global()
            axis.coastlines(linewidth=0.55)
            axis.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.45)
            axis.set_title(f"{scwd_comparison_detail(item)}  |  SCWD={item['scwd']:.4g}", fontsize=10)
        for axis in flat_axes[len(series):]:
            axis.set_visible(False)
        figure.suptitle(f"SCWD anchor contributions: {label} ({comparison_kind})", fontsize=14)
        figure.subplots_adjust(left=0.03, right=0.87, bottom=0.03, top=0.88, wspace=0.08, hspace=0.24)
        figure.colorbar(
            image, ax=flat_axes[:len(series)].tolist(), fraction=0.035, pad=0.035,
            label="Local SCWD magnitude",
        )
        kind_root = scwd_plot_root(output_root, comparison_kind)
        output_path = kind_root / f"{scwd_output_stem(series[0])}.png"
        save_figure_bundle(figure, output_path, plot_type="baseline_plot", dpi=220, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved SCWD anchor map to: {output_path}")


def plot_scwd_top_w1_distributions(diagnostics, cfg, output_root):
    """Overlay candidate and test-ERA5 response distributions at the top local-W1 anchors."""
    if not diagnostics:
        return
    n_bins = max(2, int(baseline_get(cfg, "scwd_distribution_bins", 40)))

    for diagnostic in diagnostics:
        distributions = diagnostic["top_w1_distributions"]
        if not distributions:
            continue
        n_cols = min(2, len(distributions))
        n_rows = int(np.ceil(len(distributions) / n_cols))
        figure, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.5 * n_rows), squeeze=False)
        flat_axes = axes.ravel()
        for axis, item in zip(flat_axes, distributions):
            values = np.concatenate([item["candidate"], item["reference"]])
            edges = np.histogram_bin_edges(values, bins=n_bins)
            axis.hist(
                item["reference"],
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.8,
                color="black",
                label=("ERA5 days 1–15" if scwd_comparison_kind(diagnostic) == "null" else "ERA5 test"),
            )
            axis.hist(
                item["candidate"],
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.8,
                color="tab:red",
                label=diagnostic["label"],
            )
            axis.set_title(
                f"{item['latitude']:.1f}°, {item['longitude']:.1f}°\nlocal W1={item['w1']:.4g}",
                fontsize=10,
            )
            axis.set_xlabel("SCWD filter response")
            axis.set_ylabel("Density")
            axis.grid(True, alpha=0.25)
        for axis in flat_axes[len(distributions):]:
            axis.set_visible(False)
        flat_axes[0].legend(fontsize=9)
        detail = scwd_comparison_detail(diagnostic)
        figure.suptitle(
            f"Highest local-W1 SCWD responses: {diagnostic['label']} ({detail})",
            fontsize=14,
        )
        figure.tight_layout(rect=[0, 0, 1, 0.94])
        kind_root = scwd_plot_root(output_root, scwd_comparison_kind(diagnostic), top_w1=True)
        suffix = f"_{int(diagnostic['lead_hour']):03d}h" if scwd_comparison_kind(diagnostic) == "forecast" else ""
        output_path = kind_root / f"{scwd_output_stem(diagnostic)}{suffix}.png"
        payload = {}
        for index, item in enumerate(distributions):
            payload[f"candidate_responses_{index}"] = item["candidate"]
            payload[f"reference_responses_{index}"] = item["reference"]
            payload[f"bin_edges_{index}"] = np.histogram_bin_edges(
                np.concatenate([item["candidate"], item["reference"]]), bins=n_bins,
            )
            payload[f"anchor_latitude_{index}"] = np.asarray(item["latitude"])
            payload[f"anchor_longitude_{index}"] = np.asarray(item["longitude"])
        save_figure_bundle(
            figure, output_path, plot_type="scwd_top_w1_distributions",
            payload=payload, dpi=220, bbox_inches="tight",
        )
        plt.close(figure)
        print(f"Saved top-W1 SCWD response distributions to: {output_path}")


def plot_scwd_mean_response_differences(diagnostics, output_root):
    """Plot candidate-minus-reference mean SCWD filter responses by anchor."""
    if not diagnostics:
        return
    values = np.stack([item["anchor_mean_response_difference"] for item in diagnostics])
    vmax = max(float(np.nanpercentile(np.abs(values), 99)), 1e-12)
    labels = sorted({(scwd_comparison_kind(item), item["label"]) for item in diagnostics})

    for comparison_kind, label in labels:
        series = sorted(
            [item for item in diagnostics
             if scwd_comparison_kind(item) == comparison_kind and item["label"] == label],
            key=lambda item: item["lead_hour"],
        )
        n_cols = min(2, len(series))
        n_rows = int(np.ceil(len(series) / n_cols))
        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.2 * n_cols, 3.6 * n_rows),
            squeeze=False,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        flat_axes = axes.ravel()
        image = None
        for axis, item in zip(flat_axes, series):
            image = axis.pcolormesh(
                item["anchor_longitudes"],
                item["anchor_latitudes"],
                item["anchor_mean_response_difference"],
                shading="auto",
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            axis.set_global()
            axis.coastlines(linewidth=0.55)
            axis.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.45)
            axis.set_title(scwd_comparison_detail(item), fontsize=10)
        for axis in flat_axes[len(series):]:
            axis.set_visible(False)
        figure.suptitle(f"Mean SCWD filter-response difference: {label} ({comparison_kind})", fontsize=14)
        figure.subplots_adjust(left=0.03, right=0.87, bottom=0.03, top=0.88, wspace=0.08, hspace=0.24)
        figure.colorbar(
            image, ax=flat_axes[:len(series)].tolist(), fraction=0.035, pad=0.035,
            label="Mean SCWD response difference (candidate − reference)",
        )
        kind_root = scwd_plot_root(output_root, comparison_kind)
        output_path = kind_root / f"{scwd_output_stem(series[0])}_mean_response_difference.png"
        save_figure_bundle(figure, output_path, plot_type="baseline_plot", dpi=220, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved mean SCWD response-difference map to: {output_path}")


def evaluate_lead_metrics(cfg, truth_ds, variables, metric_names, temporary_dir):
    """Score forecasts against ERA5 at their exact valid timestamps."""
    results = []
    scwd_diagnostics = []
    global_mean_diagnostics = []
    max_samples = int(baseline_get(cfg, "eval_samples", cfg_get(cfg, "max_samples", 100)))
    variable_name = joint_variable_name(variables)
    configured_leads = {int(value) for value in cfg.lead_times}
    forecast_files = baseline_get(cfg, "forecast_files", None)
    if not forecast_files:
        raise ValueError("baseline.forecast_files must contain at least one named forecast file.")

    for label, configured_paths in forecast_files.items():
        opened = [normalize_prediction_timedelta(safe_open_dataset(path)) for path in configured_paths]
        forecast_ds = select_level(concatenate_forecasts(opened), cfg.get("level"))
        if forecast_ds.sizes.get("time", 0) == 0:
            print(f"Skipping {label}: forecast file contains no samples.")
            forecast_ds.close()
            continue

        missing_variables = [variable for variable in variables if variable not in forecast_ds.data_vars]
        if missing_variables:
            print(f"Skipping {label}: missing variables {missing_variables}")
            forecast_ds.close()
            continue

        available_leads = lead_hours(forecast_ds)
        lead_pairs = [
            (lead_idx, int(lead_hour))
            for lead_idx, lead_hour in enumerate(available_leads)
            if int(lead_hour) in configured_leads
        ]
        missing_leads = sorted(configured_leads - {lead_hour for _, lead_hour in lead_pairs})
        if missing_leads:
            print(f"Warning: {label} is missing configured lead hours: {missing_leads}")
        if not lead_pairs:
            print(f"Skipping {label}: none of the configured lead hours are available.")
            forecast_ds.close()
            continue

        for lead_idx, lead_hour in tqdm(lead_pairs, desc=f"{label} lead distribution metrics"):
            pairs = [
                pair for pair in forecast_pairs(
                    forecast_ds, truth_ds, cfg, "test", [lead_hour]
                ) if pair.lead_index == lead_idx
            ]
            pairs = evenly_spaced_pairs(pairs, max_samples)
            if not pairs:
                print(f"Skipping {label} +{lead_hour}h: no exact monthly test pairs.")
                continue
            time_indices = [pair.forecast_index for pair in pairs]
            reference_ds = truth_ds.isel(time=[pair.era5_index for pair in pairs])
            reference_indices = list(range(len(pairs)))
            reference_stats = streaming_reference_stats(
                cfg, reference_ds, variables, reference_indices
            )
            reference_features = streaming_joint_features(
                cfg, reference_ds, variables, reference_stats, reference_indices,
                metric_names, Path(temporary_dir) / "forecast-reference-scwd.dat",
                description=f"{label} +{lead_hour}h exact ERA5 reference",
            )
            candidate_features = streaming_joint_features(
                cfg,
                forecast_ds,
                variables,
                reference_stats,
                time_indices,
                metric_names,
                Path(temporary_dir) / "forecast-candidate-scwd.dat",
                lead_idx=lead_idx,
                description=f"{label} +{lead_hour}h features",
            )
            metrics = selected_distribution_metric_values(
                candidate_features, reference_features, metric_names, cfg
            )
            row = {
                "label": label,
                "variable": variable_name,
                "lead_hour": lead_hour,
                "n_samples": metrics["n_samples"],
                "pairwise_n_samples": metrics["pairwise_n_samples"],
                "is_null": False,
            }
            row.update(coverage_metadata(pairs))
            row.update({metric_name: metrics[metric_name] for metric_name in metric_names})
            if "scwd" in metric_names:
                scwd_diagnostics.append(
                    scwd_anchor_diagnostic(
                        candidate_features,
                        reference_features,
                        cfg,
                        label,
                        lead_hour,
                        metrics["scwd"],
                    )
                )
            if "global_mean_wasserstein" in metric_names:
                global_mean_diagnostics.append(
                    global_mean_wasserstein_diagnostic(
                        candidate_features, reference_features, cfg, label, lead_hour,
                        metrics["global_mean_wasserstein"],
                    )
                )
            if row["n_samples"] == 0:
                print(f"Warning: no valid joint samples for {label} lead={lead_hour} variables={variables}")
            results.append(row)
            close_feature_memmaps(candidate_features)
            close_feature_memmaps(reference_features)
        forecast_ds.close()

    return results, scwd_diagnostics, global_mean_diagnostics


def corruption_types_from_config(cfg):
    """Return every corruption configured specifically for baseline probes."""
    return [str(value) for value in baseline_get(cfg, "corruptions", [])]


def is_u_wind_variable(variable):
    """Return true when a variable name looks like a 10m U-wind component."""
    normalized = str(variable).lower()
    return (
        "10m_u_component" in normalized
        or normalized in {"u10", "10u", "u_component_of_wind"}
        or normalized.endswith("_u_component_of_wind")
    )


def is_v_wind_variable(variable):
    """Return true when a variable name looks like a 10m V-wind component."""
    normalized = str(variable).lower()
    return (
        "10m_v_component" in normalized
        or normalized in {"v10", "10v", "v_component_of_wind"}
        or normalized.endswith("_v_component_of_wind")
    )


def can_apply_wind_vector_corruption(variables):
    """Return whether variables match the channel layout used by wind corruptions."""
    return (
        len(variables) > V10_CHANNEL
        and is_u_wind_variable(variables[U10_CHANNEL])
        and is_v_wind_variable(variables[V10_CHANNEL])
    )


def compatible_corruption_types(corruption_types, variables):
    """Drop standard metric corruption probes that are incompatible with variables."""
    fixed_wind_corruptions = {"wind_patch_shuffle", "wind_shuffled", "wind_rotation", "wind_rotated"}
    compatible = []
    skipped = []
    for corruption_type in corruption_types:
        if corruption_type in fixed_wind_corruptions and not can_apply_wind_vector_corruption(variables):
            skipped.append(corruption_type)
            continue
        compatible.append(corruption_type)
    if skipped:
        print(
            "Skipping baseline corruption(s) incompatible with variables "
            f"{variables}: {', '.join(skipped)}"
        )
    return compatible


def corruption_levels(corruption_type, cfg):
    """Return severity levels for one train-time corruption."""
    n_steps = int(baseline_get(cfg, "corruption_steps", 7))
    max_severity = corruption_max_severity(corruption_type, cfg)
    if n_steps <= 1:
        return [max_severity]
    return np.linspace(0.0, max_severity, n_steps).tolist()


def corruption_max_severity(corruption_type, cfg):
    """Return the configured maximum, honoring per-corruption overrides."""
    maximum = baseline_get(cfg, "corruption_severity_max", cfg_get(cfg, "corruption_severity_max", 1.0))
    overrides = baseline_get(cfg, "corruption_severity_max_overrides", {}) or {}
    return float(overrides.get(str(corruption_type), maximum))


def representative_corruption_time_index(time_indices, configured_index=None):
    """Choose a reproducible sample from the same times used by the corruption sweep."""
    if not time_indices:
        raise ValueError("No timestamps available for a corruption visualization.")
    if configured_index is not None:
        index = int(configured_index)
        if index not in time_indices:
            raise ValueError(
                "baseline.corruption_visualization_time_index must be one of the "
                "timestamps sampled by the corruption sweep."
            )
        return index
    return int(time_indices[len(time_indices) // 2])


def normalization_stats_for_corruptions(cfg, truth_ds, variables):
    """Compute normalization stats required by training-time corruptions."""
    stats_ds = select_level(
        select_era5_split(truth_ds, cfg, "train", coverage="corruption"), cfg.get("level")
    )
    means = {}
    stds = {}
    for variable in variables:
        means[variable] = float(stats_ds[variable].mean())
        std = float(stats_ds[variable].std())
        stds[variable] = std if std > 1e-8 else 1.0
    return means, stds


def evaluate_corruption_metrics(
    cfg, truth_ds, _reference_stats, variables, metric_names, temporary_dir,
    return_scwd_diagnostics=False,
):
    """Compare corrupted test fields to matched clean test ERA5."""
    results = []
    scwd_diagnostics = []
    max_samples = int(
        baseline_get(
            cfg,
            "corruption_eval_samples",
            baseline_get(cfg, "eval_samples", cfg_get(cfg, "max_samples", 100)),
        )
    )
    variable_name = joint_variable_name(variables)
    test_ds = select_level(
        select_era5_split(truth_ds, cfg, "test", coverage="corruption"), cfg.get("level")
    )
    test_indices = sample_time_indices(test_ds, max_samples)
    if not test_indices:
        raise ValueError("No ERA5 test samples available for corruption probes.")
    train_ds = select_level(
        select_era5_split(truth_ds, cfg, "train", coverage="corruption"), cfg.get("level")
    )
    null_ds = select_level(
        select_era5_split(truth_ds, cfg, "null", coverage="corruption"), cfg.get("level")
    )
    train_indices = sample_time_indices(train_ds, max_samples)
    if not train_indices:
        raise ValueError("No ERA5 train samples available for corruption probes.")
    null_indices = sample_time_indices(null_ds, max_samples)
    reference_stats = streaming_reference_stats(cfg, test_ds, variables, test_indices)
    null_reference_stats = streaming_reference_stats(cfg, train_ds, variables, train_indices)
    test_features = streaming_joint_features(
        cfg,
        test_ds,
        variables,
        reference_stats,
        test_indices,
        metric_names,
        Path(temporary_dir) / "corruption-test-scwd.dat",
        description="clean ERA5 test corruption features",
    )
    train_features = streaming_joint_features(
        cfg, train_ds, variables, null_reference_stats,
        train_indices,
        metric_names,
        Path(temporary_dir) / "corruption-train-scwd.dat",
        description="ERA5 train corruption-reference features",
    )
    null_features = streaming_joint_features(
        cfg, null_ds, variables, null_reference_stats, null_indices, metric_names,
        Path(temporary_dir) / "corruption-null-scwd.dat",
        description="ERA5 second-half null features",
    )

    for corruption_type in compatible_corruption_types(corruption_types_from_config(cfg), variables):
        levels = corruption_levels(corruption_type, cfg)
        full_severity = float(max(levels))
        zero_metrics = selected_distribution_metric_values(test_features, test_features, metric_names, cfg)
        zero_row = {
            "label": "ERA5 test matched reference",
            "variable": variable_name,
            "corruption": corruption_type,
            "severity": 0.0,
            "n_samples": zero_metrics["n_samples"],
            "pairwise_n_samples": zero_metrics["pairwise_n_samples"],
            "is_null": False,
        }
        zero_row.update({metric_name: zero_metrics[metric_name] for metric_name in metric_names})
        results.append(zero_row)
        null_metrics = selected_distribution_metric_values(
            null_features, train_features, metric_names, cfg
        )
        null_row = {
            "label": ERA5_NULL_LABEL,
            "variable": variable_name,
            "corruption": corruption_type,
            "severity": 0.0,
            "n_samples": null_metrics["n_samples"],
            "pairwise_n_samples": null_metrics["pairwise_n_samples"],
            "is_null": True,
        }
        null_row.update({metric_name: null_metrics[metric_name] for metric_name in metric_names})
        results.append(null_row)
        for severity in tqdm(levels, desc=f"{corruption_type} distribution metrics"):
            if float(severity) == 0.0:
                continue
            candidate_features = streaming_joint_features(
                cfg,
                test_ds,
                variables,
                reference_stats,
                test_indices,
                metric_names,
                Path(temporary_dir) / "corruption-candidate-scwd.dat",
                corruption_type=corruption_type,
                severity=severity,
                description=f"{corruption_type} severity={severity:.3g}",
            )
            base_row = {
                "label": "ERA5 test corrupted vs matched test",
                "variable": variable_name,
                "corruption": corruption_type,
                "severity": float(severity),
                "is_null": False,
            }
            metrics = selected_distribution_metric_values(
                candidate_features, test_features, metric_names, cfg
            )
            row = dict(base_row)
            row.update({metric_name: metrics[metric_name] for metric_name in metric_names})
            row["n_samples"] = metrics["n_samples"]
            row["pairwise_n_samples"] = metrics["pairwise_n_samples"]
            if "scwd" in metric_names and np.isclose(float(severity), full_severity):
                scwd_diagnostics.append(scwd_anchor_diagnostic(
                    candidate_features, test_features, cfg, corruption_type, 0, metrics["scwd"],
                    comparison_kind="corruption", severity=float(severity),
                ))
            if row["n_samples"] == 0:
                print(
                    "Warning: no valid joint samples for "
                    f"corruption={corruption_type} severity={severity} variables={variables}"
                )
            results.append(row)
            close_feature_memmaps(candidate_features)

    close_feature_memmaps(test_features)
    close_feature_memmaps(train_features)
    close_feature_memmaps(null_features)
    return (results, scwd_diagnostics) if return_scwd_diagnostics else results


def baseline_output_dir(cfg, variables):
    """Return the namespaced output directory for one baseline variant."""
    return Path(str(baseline_get(cfg, "output_dir"))) / variable_tag(variables)


def write_metric_csv(rows, metric_names, output_path, experiment):
    """Write one experiment type with a purpose-specific schema."""
    if experiment == "lead_time":
        fieldnames = [
            "label", "variable", "lead_hour", "is_null", "n_samples",
            "pairwise_n_samples", "n_pairs", "initialization_start",
            "initialization_end", "valid_start", "valid_end",
        ] + metric_names
    elif experiment == "corruption_strength":
        fieldnames = [
            "label", "variable", "corruption", "severity", "is_null", "n_samples", "pairwise_n_samples"
        ] + metric_names
    else:
        raise ValueError(f"Unknown baseline experiment: {experiment}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {experiment} metric values to: {output_path}")


def read_metric_csv(output_path, metric_names, experiment):
    """Read persisted metric rows and restore the numeric types used by plotting."""
    if not output_path.exists():
        raise FileNotFoundError(
            f"Missing baseline evaluation artifact: {output_path}. "
            "Run scripts/evaluate_standard_metric_baselines.py first."
        )
    with open(output_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = set(metric_names) | {"n_samples", "pairwise_n_samples"}
    numeric.add("lead_hour" if experiment == "lead_time" else "severity")
    for row in rows:
        for field in numeric:
            if field in row and row[field] != "":
                row[field] = float(row[field])
        for field in ("n_samples", "pairwise_n_samples", "lead_hour"):
            if field in row and row[field] != "":
                row[field] = int(float(row[field]))
    return rows


def row_is_null(row):
    return str(row.get("is_null", False)).lower() == "true"

ABSOLUTE_DISPLAY_METRICS = {"mean_bias", "std_ratio_error"}


def display_metric_value(row, metric_name):
    """Return a presentation-only metric value for baseline figures."""
    value = float(row.get(metric_name, np.nan))
    return abs(value) if metric_name in ABSOLUTE_DISPLAY_METRICS else value


def displayed_metric_name(metric_name):
    return f"|{metric_name}|" if metric_name in ABSOLUTE_DISPLAY_METRICS else metric_name


def relative_corruption_coordinates(rows, value_key):
    """Map a corruption-specific native range onto the common visual [0, 1] range."""
    values = [float(row[value_key]) for row in rows]
    maximum = max(values, default=0.0)
    if maximum <= 0.0:
        return [0.0 for _ in values]
    return [value / maximum for value in values]


def corruption_range_label(label, rows, value_key):
    maximum = max((float(row[value_key]) for row in rows), default=0.0)
    return f"{label} [0, {maximum:g}]"


def metric_normalization_scales(rows, metric_names):
    """Return one robust, global-per-figure normalization factor per metric."""
    scales = {}
    for metric_name in metric_names:
        values = np.asarray([display_metric_value(row, metric_name) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        positive = values[values > 0.0]
        scale = positive.max() if positive.size else (np.abs(values).max() if values.size else 0.0)
        scales[metric_name] = float(scale) if scale > 0.0 else 1.0
    return scales


def normalized_metric_value(row, metric_name, scales):
    """Normalize one presentation metric value by its global figure scale."""
    value = display_metric_value(row, metric_name)
    return value / scales[metric_name] if np.isfinite(value) else np.nan


def metric_colors(metric_names):
    return {
        metric_name: color
        for metric_name, color in zip(
            metric_names, plt.cm.tab10(np.linspace(0, 1, max(len(metric_names), 1)))
        )
    }


def plot_normalized_lead_metrics_by_model(rows, metric_names, variables, output_root):
    """Plot one model per panel and globally normalized metrics per line."""
    metric_names = plotted_metric_names(metric_names)
    if not metric_names:
        return
    variable = joint_variable_name(variables)
    variable_rows = [row for row in rows if row["variable"] == variable]
    era5_shift_label = ERA5_NULL_LABEL
    labels = sorted({row["label"] for row in variable_rows if row["label"] != era5_shift_label})
    if not labels:
        return
    scales, colors = metric_normalization_scales(variable_rows, metric_names), metric_colors(metric_names)
    n_cols = min(2, len(labels)); n_rows = int(np.ceil(len(labels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.8 * n_rows), squeeze=False)
    era5_rows = [row for row in variable_rows if row["label"] == era5_shift_label]
    for axis, label in zip(axes.ravel(), labels):
        series = sorted([row for row in variable_rows if row["label"] == label], key=lambda row: row["lead_hour"])
        for metric_index, metric_name in enumerate(metric_names):
            axis.plot(
                [row["lead_hour"] for row in series],
                [normalized_metric_value(row, metric_name, scales) for row in series],
                marker=series_marker(metric_index), linewidth=1.6, color=colors[metric_name], label=metric_name,
            )
            if era5_rows:
                axis.scatter(
                    [0], [normalized_metric_value(era5_rows[0], metric_name, scales)],
                    marker="D", s=26, color=colors[metric_name], zorder=4,
                )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        axis.set(title=label, xlabel="Lead time (hours)")
        axis.grid(True, alpha=0.3)
    for axis in axes.ravel()[len(labels):]:
        axis.axis("off")
    fig.supylabel("Normalized divergence (metric maximum = 1)")
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle(f"Normalized Distributional Metrics by Forecast Model: {variable}\nDiamonds: ERA5 days 16–end vs days 1–15", fontsize=14)
    fig.tight_layout(rect=[0.03, 0, 0.82, 0.91])
    output_path = output_root / "plots" / "lead_time_by_model_normalized.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(fig, output_path, plot_type="baseline_plot", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved normalized lead-time-by-model plot to: {output_path}")
def plot_normalized_corruption_metrics_by_type(rows, metric_names, variables, output_root):
    """Plot one corruption per panel and globally normalized metrics per line."""
    metric_names = plotted_metric_names(metric_names)
    if not metric_names:
        return
    metric_names = plotted_metric_names(metric_names)
    """Plot one corruption per panel and globally normalized metrics per line."""
    variable = joint_variable_name(variables)
    variable_rows = [row for row in rows if row["variable"] == variable]
    corruptions = sorted({row["corruption"] for row in variable_rows})
    if not corruptions:
        return
    scales, colors = metric_normalization_scales(variable_rows, metric_names), metric_colors(metric_names)
    n_cols = 2; n_rows = int(np.ceil(len(corruptions) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.8 * n_rows), squeeze=False)
    for axis, corruption in zip(axes.ravel(), corruptions):
        series = sorted(
            [row for row in variable_rows if row["corruption"] == corruption and not row_is_null(row)],
            key=lambda row: row["severity"],
        )
        null_row = next((row for row in variable_rows if row["corruption"] == corruption and row_is_null(row)), None)
        for metric_index, metric_name in enumerate(metric_names):
            axis.plot(
                [row["severity"] for row in series],
                [normalized_metric_value(row, metric_name, scales) for row in series],
                marker=series_marker(metric_index), linewidth=1.6, color=colors[metric_name], label=metric_name,
            )
            if null_row is not None:
                axis.scatter(
                    [0], [normalized_metric_value(null_row, metric_name, scales)],
                    marker="D", s=26, color=colors[metric_name], zorder=4,
                )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        axis.set(title=corruption, xlabel="Corruption severity")
        axis.grid(True, alpha=0.3)
    for axis in axes.ravel()[len(corruptions):]:
        axis.axis("off")
    fig.supylabel("Normalized divergence (metric maximum = 1)")
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle(f"Normalized Distributional Metrics by Corruption: {variable}\nDiamonds: ERA5 second-half null", fontsize=14)
    fig.tight_layout(rect=[0.03, 0, 0.82, 0.91])
    output_path = output_root / "plots" / "corruption_by_type_normalized.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(fig, output_path, plot_type="baseline_plot", dpi=220, bbox_inches="tight")
def plot_lead_metrics(rows, metric_names, variables, output_root):
    """Plot point-estimate metrics versus configured forecast lead time."""
    metric_names = plotted_metric_names(metric_names)
    if not metric_names:
        return

def plot_lead_metrics(rows, metric_names, variables, output_root):
    metric_names = plotted_metric_names(metric_names)
    """Plot point-estimate metrics versus configured forecast lead time."""
    variable = joint_variable_name(variables)
    variable_rows = [row for row in rows if row["variable"] == variable]
    if not variable_rows:
        return

    n_cols = 2
    n_rows = int(np.ceil(len(metric_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), squeeze=False)
    axes = axes.ravel()
    era5_shift_label = ERA5_NULL_LABEL
    labels = sorted({row["label"] for row in variable_rows if row["label"] != era5_shift_label})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))

    for metric_idx, metric_name in enumerate(metric_names):
        metric_label = displayed_metric_name(metric_name)
        ax = axes[metric_idx]
        for label_index, (color, label) in enumerate(zip(colors, labels)):
            series = sorted(
                [row for row in variable_rows if row["label"] == label],
                key=lambda row: row["lead_hour"],
            )
            ax.plot(
                [row["lead_hour"] for row in series],
                [display_metric_value(row, metric_name) for row in series],
                marker=series_marker(label_index), linewidth=1.8, color=color, label=label,
            )
        era5_rows = [row for row in variable_rows if row["label"] == era5_shift_label]
        if era5_rows:
            ax.plot(
                [0], [display_metric_value(era5_rows[0], metric_name)], marker="D", markersize=7,
                linestyle="None", color="black", label=era5_shift_label,
            )
        ax.set_title(metric_label)
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(metric_names):]:
        ax.axis("off")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.suptitle(f"Distributional Metrics vs Lead Time: {variable}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.82, 0.96])
    output_path = output_root / "plots" / "lead_time.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(fig, output_path, plot_type="baseline_plot", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved lead-time metric plot to: {output_path}")


def plot_corruption_metrics(rows, metric_names, variables, output_root):
    """Plot point-estimate metrics versus corruption strength."""
    metric_names = plotted_metric_names(metric_names)
    if not metric_names:
        return
    variable = joint_variable_name(variables)
    variable_rows = [row for row in rows if row["variable"] == variable]
    if not variable_rows:
        return

    corruptions = sorted({row["corruption"] for row in variable_rows})
    metric_output_dir = output_root / "plots" / "corruption"
    metric_output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in metric_names:
        metric_label = displayed_metric_name(metric_name)
        n_cols = 2
        n_rows = int(np.ceil(len(corruptions) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)
        axes = axes.flatten()
        for axis_idx, corruption_type in enumerate(corruptions):
            ax = axes[axis_idx]
            series = sorted(
                [row for row in variable_rows if row["corruption"] == corruption_type and not row_is_null(row)],
                key=lambda row: row["severity"],
            )
            ax.plot(
                [row["severity"] for row in series],
                [display_metric_value(row, metric_name) for row in series],
                marker="o", linewidth=1.8,
            )
            null_row = next((row for row in variable_rows if row["corruption"] == corruption_type and row_is_null(row)), None)
            if null_row is not None:
                ax.scatter(
                    [0.0], [display_metric_value(null_row, metric_name)], marker="D", s=42,
                    color="black", zorder=3, label="ERA5 second-half null",
                )
            ax.set_title(corruption_type)
            ax.set_xlabel("Corruption severity")
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)
            if axis_idx == 0:
                ax.legend(fontsize=8)
        for ax in axes[len(corruptions):]:
            ax.axis("off")
        fig.suptitle(
            f"{metric_label} vs Corruption Strength: {variable}\n"
            "Diamond at zero: ERA5 second-half null",
            fontsize=15,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = metric_output_dir / f"{metric_name}.png"
        save_figure_bundle(fig, output_path, plot_type="baseline_plot", dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved corruption metric plot to: {output_path}")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(corruptions), 1)))
    n_cols = 2
    n_rows = int(np.ceil(len(metric_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), squeeze=False)
    axes = axes.ravel()
    for metric_idx, metric_name in enumerate(metric_names):
        metric_label = displayed_metric_name(metric_name)
        ax = axes[metric_idx]
        for corruption_index, (color, corruption_type) in enumerate(zip(colors, corruptions)):
            series = sorted(
                [row for row in variable_rows if row["corruption"] == corruption_type and not row_is_null(row)],
                key=lambda row: row["severity"],
            )
            ax.plot(
                relative_corruption_coordinates(series, "severity"),
                [display_metric_value(row, metric_name) for row in series],
                marker=series_marker(corruption_index), linewidth=1.8, color=color, label=corruption_range_label(corruption_type, series, "severity"),
            )
            null_row = next((row for row in variable_rows if row["corruption"] == corruption_type and row_is_null(row)), None)
            if null_row is not None:
                ax.scatter(
                    [0.0], [display_metric_value(null_row, metric_name)], marker="D", s=36,
                    color="black", zorder=3,
                    label=(ERA5_NULL_LABEL if metric_idx == 0 and corruption_type == corruptions[0] else "_nolegend_"),
                )
        ax.set_title(metric_label)
        ax.set_xlabel("Relative corruption severity (1 = corruption-specific maximum)")
        ax.set_ylabel(metric_label)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(metric_names):]:
        ax.axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle(
        f"Distributional Metrics vs Corruption Strength: {variable}\n"
        "Diamond at zero: ERA5 second-half null",
        fontsize=15,
    )
    fig.tight_layout(rect=[0.03, 0, 0.82, 0.96])
    output_path = output_root / "plots" / "corruption_strength.png"
    save_figure_bundle(fig, output_path, plot_type="baseline_plot", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined corruption metric plot to: {output_path}")


def evaluate_corruption_disturbances(cfg, truth_ds, normalization_stats, variables, output_root):
    """Persist physical perturbations for cheap, repeatable gallery plotting."""
    means, stds = normalization_stats
    output_path = output_root / "data" / "corruption_disturbances.nc"
    if not bool(baseline_get(cfg, "plot_corruption_disturbances", True)):
        output_path.unlink(missing_ok=True)
        return
    max_samples = int(
        baseline_get(
            cfg, "corruption_eval_samples", baseline_get(cfg, "eval_samples", cfg_get(cfg, "max_samples", 100))
        )
    )
    eval_ds = select_level(
        select_era5_split(truth_ds, cfg, "test", coverage="corruption"), cfg.get("level")
    )
    time_indices = sample_time_indices(eval_ds, max_samples)
    sample_index = representative_corruption_time_index(
        time_indices, baseline_get(cfg, "corruption_visualization_time_index", None)
    )
    sample = eval_ds.isel(time=sample_index)
    latitudes = np.asarray(sample[variables[0]].latitude.values, dtype=np.float64)
    longitudes = np.asarray(sample[variables[0]].longitude.values, dtype=np.float64)
    clean = np.stack(
        [canonical_latlon(sample[variable].values, latitudes).astype(np.float32) for variable in variables]
    )
    standardized = np.stack(
        [
            (clean[channel] - means[variable]) / stds[variable]
            for channel, variable in enumerate(variables)
        ]
    ).astype(np.float32)
    timestamp = str(np.asarray(sample.time.values))[:19]
    base_seed = int(baseline_get(cfg, "corruption_seed", 0))
    corruption_names = []
    severity_values = []
    disturbance_values = []
    corrupted_field_values = []

    for corruption_type in compatible_corruption_types(corruption_types_from_config(cfg), variables):
        levels = corruption_levels(corruption_type, cfg)
        special_corruption = (
            corruption_type in STRUCTURED_NEAR_NULL_CORRUPTIONS | DATA_DEPENDENT_CORRUPTIONS
        )
        donor_standardized = None
        if corruption_type == "hemisphere_splice":
            sample_position = time_indices.index(sample_index)
            donor_positions = deranged_sample_positions(len(time_indices), base_seed)
            donor_index = time_indices[int(donor_positions[sample_position])]
            donor_sample = eval_ds.isel(time=donor_index)
            donor_clean = np.stack(
                [
                    canonical_latlon(donor_sample[variable].values, latitudes).astype(np.float32)
                    for variable in variables
                ]
            )
            donor_standardized = np.stack(
                [
                    (donor_clean[channel] - means[variable])
                    / stds[variable]
                    for channel, variable in enumerate(variables)
                ]
            ).astype(np.float32)
        corrupted_fields = []
        disturbances = []
        for severity in levels:
            if special_corruption:
                corrupted = apply_special_baseline_corruption(
                    standardized, corruption_type, severity, latitudes, cfg, donor_standardized,
                    random_seed=corruption_sample_seed(base_seed, corruption_type, sample_index),
                )
            else:
                seed = corruption_sample_seed(base_seed, corruption_type, sample_index)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    corrupted = apply_configured_corruption(
                        torch.from_numpy(standardized), corruption_type, float(severity)
                    ).detach().cpu().numpy()
            corrupted_physical = np.stack(
                [
                    corrupted[channel] * stds[variable] + means[variable]
                    for channel, variable in enumerate(variables)
                ]
            )
            corrupted_fields.append(corrupted_physical)
            disturbances.append(
                np.stack(
                    [
                        (corrupted[channel] - standardized[channel]) * stds[variable]
                        for channel, variable in enumerate(variables)
                    ]
                )
            )
        disturbances = np.asarray(disturbances)
        corruption_names.append(corruption_type)
        severity_values.append(np.asarray(levels, dtype=np.float64))
        disturbance_values.append(disturbances.astype(np.float32))
        corrupted_field_values.append(np.asarray(corrupted_fields, dtype=np.float32))

    if not disturbance_values:
        output_path.unlink(missing_ok=True)
        return
    dataset = xr.Dataset(
        data_vars={
            "disturbance": (
                ("corruption", "severity_index", "variable", "latitude", "longitude"),
                np.stack(disturbance_values),
            ),
            "corrupted_field": (
                ("corruption", "severity_index", "variable", "latitude", "longitude"),
                np.stack(corrupted_field_values),
            ),
            "severity": (("corruption", "severity_index"), np.stack(severity_values)),
        },
        coords={
            "corruption": corruption_names,
            "variable": variables,
            "latitude": latitudes,
            "longitude": longitudes,
        },
        attrs={
            "timestamp": timestamp,
            "description": "Physical corrupted fields and their differences from clean ERA5 for one representative sample.",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    dataset.close()
    print(f"Saved corruption disturbance data to: {output_path}")


def plot_corruption_field_gallery(
    output_dir, corruption_type, levels, fields, variables, latitudes, longitudes, timestamp,
    title, colorbar_label, filename, symmetric=False,
):
    """Render one two-row-per-field corruption gallery from persisted physical values."""
    if symmetric:
        upper = np.maximum(np.nanpercentile(np.abs(fields), 99, axis=(0, 2, 3)), 1e-8)
        lower = -upper
        cmap = "RdBu_r"
    else:
        lower = np.nanpercentile(fields, 1, axis=(0, 2, 3))
        upper = np.nanpercentile(fields, 99, axis=(0, 2, 3))
        equal = np.isclose(lower, upper)
        lower[equal] -= 1e-8
        upper[equal] += 1e-8
        cmap = "viridis"
    severity_rows = 2
    severity_columns = int(np.ceil(len(levels) / severity_rows))
    figure, axes = plt.subplots(
        len(variables) * severity_rows, severity_columns,
        figsize=(3.5 * severity_columns, 3.15 * len(variables) * severity_rows),
        squeeze=False, subplot_kw={"projection": ccrs.PlateCarree()},
    )
    for variable_index, variable in enumerate(variables):
        row_start = variable_index * severity_rows
        image = None
        for severity_index, severity in enumerate(levels):
            row = row_start + severity_index // severity_columns
            column = severity_index % severity_columns
            axis = axes[row, column]
            image = axis.pcolormesh(
                longitudes, latitudes, fields[severity_index, variable_index], shading="auto",
                cmap=cmap, vmin=float(lower[variable_index]), vmax=float(upper[variable_index]),
                transform=ccrs.PlateCarree(),
            )
            axis.set_global()
            axis.coastlines(linewidth=0.5)
            axis.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            axis.set_title(f"severity={severity:.3g}", fontsize=9)
            if column == 0:
                axis.set_ylabel(variable.replace("_", " "), fontsize=9)
        for severity_index in range(len(levels), severity_rows * severity_columns):
            row = row_start + severity_index // severity_columns
            column = severity_index % severity_columns
            axes[row, column].set_visible(False)
        figure.colorbar(
            image, ax=axes[row_start:row_start + severity_rows, :].ravel().tolist(),
            fraction=0.035, pad=0.025, label=colorbar_label,
        )
    figure.suptitle(f"{corruption_type}: {title} at each severity\nERA5 {timestamp}", fontsize=14)
    figure.subplots_adjust(left=0.03, right=0.87, bottom=0.03, top=0.88, wspace=0.07, hspace=0.25)
    output_path = output_dir / filename
    save_figure_bundle(figure, output_path, plot_type="baseline_plot", dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved corruption gallery to: {output_path}")



def plot_combined_corruption_gallery(output_dir, corruptions, levels, corrupted_fields, disturbances,
                                     variables, latitudes, longitudes, timestamp):
    """Render a large raw-field/difference overview with one pair of rows per corruption."""
    n_corruptions, n_severity, n_variables = corrupted_fields.shape[:3]
    n_columns = n_severity
    for variable_index, variable in enumerate(variables):
        raw = corrupted_fields[:, :, variable_index]
        difference = disturbances[:, :, variable_index]
        raw_low, raw_high = np.nanpercentile(raw, [1.0, 99.0])
        if np.isclose(raw_low, raw_high):
            raw_low -= 1e-8; raw_high += 1e-8
        difference_limit = max(float(np.nanpercentile(np.abs(difference), 99.0)), 1e-8)
        n_rows = 2 * n_corruptions
        figure, axes = plt.subplots(
            n_rows, n_columns,
            figsize=(3.05 * n_columns, 2.45 * n_rows + 0.8),
            squeeze=False, subplot_kw={"projection": ccrs.PlateCarree()},
        )
        raw_artist = difference_artist = None
        for corruption_index, corruption in enumerate(corruptions):
            for severity_index in range(n_severity):
                raw_axis = axes[2 * corruption_index, severity_index]
                difference_axis = axes[2 * corruption_index + 1, severity_index]
                raw_artist = raw_axis.pcolormesh(
                    longitudes, latitudes, raw[corruption_index, severity_index], shading="auto",
                    cmap="viridis", vmin=raw_low, vmax=raw_high, transform=ccrs.PlateCarree(),
                )
                difference_artist = difference_axis.pcolormesh(
                    longitudes, latitudes, difference[corruption_index, severity_index], shading="auto",
                    cmap="RdBu_r", vmin=-difference_limit, vmax=difference_limit,
                    transform=ccrs.PlateCarree(),
                )
                for axis in (raw_axis, difference_axis):
                    axis.set_global(); axis.coastlines(linewidth=0.42)
                    axis.add_feature(cfeature.BORDERS, linewidth=0.25, alpha=0.4)
                raw_axis.set_title(f"severity={levels[corruption_index, severity_index]:.3g}", fontsize=8)
            axes[2 * corruption_index, 0].set_ylabel(f"{corruption}\ncorrupted", fontsize=8)
            axes[2 * corruption_index + 1, 0].set_ylabel(f"{corruption}\nminus ERA5", fontsize=8)
        figure.suptitle(
            f"All corruption probes: {variable.replace('_', ' ')}\nERA5 {timestamp}", fontsize=15,
        )
        figure.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.075, wspace=0.025, hspace=0.16)
        raw_colorbar_axis = figure.add_axes([0.10, 0.022, 0.34, 0.012])
        difference_colorbar_axis = figure.add_axes([0.58, 0.022, 0.34, 0.012])
        figure.colorbar(raw_artist, cax=raw_colorbar_axis, orientation="horizontal", label="Corrupted field")
        figure.colorbar(difference_artist, cax=difference_colorbar_axis, orientation="horizontal", label="Corrupted − ERA5")
        suffix = "all_corruptions_gallery.png" if n_variables == 1 else f"all_corruptions_gallery_{variable}.png"
        output_path = output_dir / suffix
        save_figure_bundle(figure, output_path, plot_type="corruption_gallery", dpi=160, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved combined corruption gallery to: {output_path}")

def plot_corruption_disturbances(output_root):
    """Render raw corrupted fields and their ERA5 differences from the saved artifact."""
    input_path = output_root / "data" / "corruption_disturbances.nc"
    if not input_path.exists():
        return
    output_dir = output_root / "plots" / "corruption" / "disturbances"
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(input_path) as dataset:
        variables = [str(value) for value in dataset.variable.values]
        latitudes = np.asarray(dataset.latitude.values)
        longitudes = np.asarray(dataset.longitude.values)
        timestamp = str(dataset.attrs.get("timestamp", ""))
        has_corrupted_fields = "corrupted_field" in dataset.variables
        for corruption_index, corruption_type in enumerate(dataset.corruption.values):
            corruption_type = str(corruption_type)
            levels = np.asarray(dataset.severity.values[corruption_index])
            disturbances = np.asarray(dataset.disturbance.values[corruption_index])
            plot_corruption_field_gallery(
                output_dir, corruption_type, levels, disturbances, variables, latitudes, longitudes, timestamp,
                "physical disturbance", "Corrupted − ERA5", f"{corruption_type}.png", symmetric=True,
            )
            if has_corrupted_fields:
                corrupted_fields = np.asarray(dataset.corrupted_field.values[corruption_index])
                plot_corruption_field_gallery(
                    output_dir, corruption_type, levels, corrupted_fields, variables, latitudes, longitudes, timestamp,
                    "corrupted field", "Corrupted field", f"{corruption_type}_corrupted.png",
                )
        if has_corrupted_fields:
            plot_combined_corruption_gallery(
                output_dir,
                [str(value) for value in dataset.corruption.values],
                np.asarray(dataset.severity.values),
                np.asarray(dataset.corrupted_field.values),
                np.asarray(dataset.disturbance.values),
                variables, latitudes, longitudes, timestamp,
            )
        else:
            print(f"{input_path} lacks raw corrupted fields; rerun standard metric evaluation once.")


def discriminator_baseline_rows(cfg, real_ds, variables, output_root):
    """Re-evaluate separately trained target discriminators into baseline outputs."""
    settings = baseline_get(cfg, "discriminator", {})
    if not bool(settings.get("enabled", False)):
        return []
    try:
        from .train_target_discriminator_baselines import (
            SFNO_VARIABLES, compatible, load_sfno_encoder,
            load_sfno_probe_checkpoint, matched_statistics, real_term, score, sfno_context_settings,
        )
    except ImportError:
        from train_target_discriminator_baselines import (
            SFNO_VARIABLES, compatible, load_sfno_encoder,
            load_sfno_probe_checkpoint, matched_statistics, real_term, score, sfno_context_settings,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corruption_train = select_era5_split(real_ds, cfg, "train", coverage="corruption")
    corruption_test = select_era5_split(real_ds, cfg, "test", coverage="corruption")
    corruption_null = select_era5_split(real_ds, cfg, "null", coverage="corruption")
    checkpoint_root = Path(str(settings["checkpoint_dir"]))
    maximum = int(settings.get("evaluation_samples", 0))
    batch_size = int(settings.get("evaluation_batch_size", 32))
    discriminator_maximum = float(
        settings.get("corruption_severity_max", corruption_max_severity("hemisphere_splice", cfg))
    )
    discriminator_overrides = settings.get("corruption_severity_max_overrides", {}) or {}
    rows = []
    architectures = [("squeezenet", variables, None)]
    masked = settings.get("equator_masked_hemisphere_splice", {}) or {}
    masked_root = checkpoint_root / "squeezenet_equator_mask"
    if bool(masked.get("enabled", True)) and masked_root.exists():
        architectures.append(("squeezenet_equator_mask", variables, None))
    attention = settings.get("attention_squeezenet", {}) or {}
    attention_root = checkpoint_root / "squeezenet_attention"
    if bool(attention.get("enabled", False)) and attention_root.exists():
        architectures.append(("squeezenet_attention", variables, None))
    elif bool(attention.get("enabled", False)):
        print(
            "Skipping attention-SqueezeNet target baselines: no checkpoints under "
            f"{attention_root}."
        )
    sfno = settings.get("sfno", {}) or {}
    if bool(sfno.get("enabled", False)):
        sfno_checkpoints = [
            checkpoint_root / architecture / kind / label.replace(" ", "_") / "model.pth"
            for architecture in ("sfno_linear", "sfno_mlp")
            for kind, label in (
                [("forecast", name) for name in settings.get("forecast_files", {})]
                + [("corruption", name) for name in corruption_types_from_config(cfg)]
            )
        ]
        if any(path.exists() for path in sfno_checkpoints):
            encoder = load_sfno_encoder(cfg, device)
            architectures.extend([
                ("sfno_linear", SFNO_VARIABLES, encoder),
                ("sfno_mlp", SFNO_VARIABLES, encoder),
            ])
        else:
            print(f"Skipping SFNO target baselines: no probe checkpoints under {checkpoint_root}.")
    for architecture, input_variables, encoder in architectures:
        targets = [
            ("forecast", label, None, paths)
            for label, paths in settings.get("forecast_files", {}).items()
        ]
        targets += [
            ("corruption", name, name, None)
            for name in corruption_types_from_config(cfg) if compatible(name, input_variables)
        ]
        if architecture == "squeezenet_equator_mask":
            targets = [("corruption", "hemisphere_splice", "hemisphere_splice", None)]
        for kind, label, corruption, path in tqdm(
            targets, desc=f"Evaluating {architecture} target baselines"
        ):
            checkpoint = (
                checkpoint_root / kind / label.replace(" ", "_") / "model.pth"
                if architecture == "squeezenet" else
                checkpoint_root / architecture / kind / label.replace(" ", "_") / "model.pth"
            )
            if not checkpoint.exists():
                print(f"Skipping {architecture} {kind}/{label}: checkpoint not found at {checkpoint}")
                continue
            if architecture in {"squeezenet", "squeezenet_attention", "squeezenet_equator_mask"}:
                model_name = (
                    settings.get("model_name", "squeezenet")
                    if architecture in {"squeezenet", "squeezenet_equator_mask"} else
                    attention.get("model_name", "squeezenet_attention")
                )
                model = WeatherDiscriminator(
                    len(variables), model_name,
                    pretrained_backbone=False,
                ).to(device)
                model.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                if architecture == "squeezenet_equator_mask":
                    model.equator_mask_degrees = float(masked.get("half_width_degrees", 10.0))
                metadata = {"encoder_pretraining": ""}
                model.eval()
            else:
                model, metadata = load_sfno_probe_checkpoint(
                    checkpoint, cfg, device, encoder=encoder,
                )
                model.sfno_use_era5_context, model.sfno_target_variables = sfno_context_settings(cfg)
            train_indices = None
            if corruption:
                means = {v: float(corruption_train[v].mean()) for v in variables}
                stds = {v: max(float(corruption_train[v].std()), 1e-8) for v in variables}
                ep_dataset = corruption_train
                null_dataset = corruption_null
                candidate = corruption_test
                test_pairs = None
            else:
                opened = [
                    normalize_prediction_timedelta(safe_open_dataset(candidate_path))
                    for candidate_path in path
                ]
                candidate = concatenate_forecasts(opened)
                train_pairs = forecast_pairs(candidate, real_ds, cfg, "train", cfg.lead_times)
                test_pairs = forecast_pairs(candidate, real_ds, cfg, "test", cfg.lead_times)
                means, stds = matched_statistics(real_ds, train_pairs, variables)
                ep_dataset = real_ds
                train_indices = [pair.era5_index for pair in train_pairs]
                model_years = np.unique(
                    np.asarray(candidate.time.values).astype("datetime64[Y]").astype(int)
                )
                model_null = select_era5_split(real_ds, cfg, "null", coverage="model")
                null_years = np.asarray(model_null.time.values).astype("datetime64[Y]").astype(int)
                null_dataset = model_null.isel(
                    time=np.flatnonzero(np.isin(null_years, model_years))
                )
            ep, epse = real_term(
                model, ep_dataset, input_variables, means, stds,
                device, maximum, batch_size,
                progress_description=f"{architecture} {label}: ERA5 days 1–15 reference",
                selected_indices=train_indices,
            )
            points = [(0.0, None, None, ERA5_NULL_LABEL, None, None)]
            if corruption:
                points.append((0.0, None, 0.0, label, None, None))
                maximum_severity = float(discriminator_overrides.get(corruption, discriminator_maximum))
                levels = np.linspace(0.0, maximum_severity, int(settings["corruption_steps"]))
                points += [(float(level), None, float(level), label, None, None) for level in levels[1:]]
            else:
                for lead_index, lead in enumerate(lead_hours(candidate)):
                    selected_pairs = [pair for pair in test_pairs if pair.lead_index == lead_index]
                    selected = [pair.forecast_index for pair in selected_pairs]
                    context_selected = [pair.era5_index for pair in selected_pairs]
                    if int(lead) in set(cfg.lead_times) and selected:
                        points.append((float(lead), lead_index, None, label, selected, context_selected))
            for x, lead_index, severity, source, selected, context_selected in points:
                mean, stderr, count = score(
                    model, null_dataset if source == ERA5_NULL_LABEL else candidate,
                    input_variables, means, stds, device,
                    lead=lead_index, corruption=corruption if severity is not None else None,
                    severity=severity or 0.0, maximum=maximum, batch_size=batch_size, cfg=cfg,
                    maximum_severity=(maximum_severity if corruption else None),
                    selected_indices=selected,
                    context_ds=(real_ds if context_selected is not None and getattr(model, "sfno_use_era5_context", False) else None),
                    context_indices=context_selected,
                    progress_description=(
                        f"{architecture} {label}: {source} "
                        f"({'severity' if severity is not None else 'lead'}={x:g})"
                    ),
                )
                rows.append({
                    "architecture": architecture,
                    "input_variables": ",".join(input_variables),
                    "encoder_pretraining": metadata.get("encoder_pretraining", ""),
                    "kind": kind, "target": label, "x": x, "source": source,
                    "is_era5_test_null": source == ERA5_NULL_LABEL, "score": ep - mean,
                    "stderr": float(np.hypot(epse, stderr)), "n_samples": count,
                    "ep_train": ep,
                })
            if not corruption:
                candidate.close()
    return rows


def write_discriminator_baselines(rows, cfg, output_root):
    data_path = output_root / "data" / "discriminator_reverse_kl.csv"
    if not rows:
        data_path.unlink(missing_ok=True)
        return
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_discriminator_baselines(output_root):
    """Read target-discriminator scores without loading models or datasets."""
    data_path = output_root / "data" / "discriminator_reverse_kl.csv"
    if not data_path.exists():
        return []
    with open(data_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("x", "score", "stderr", "ep_train"):
            row[field] = float(row[field])
        row["n_samples"] = int(float(row["n_samples"]))
        row["is_era5_test_null"] = str(row["is_era5_test_null"]).lower() == "true"
    return rows


def plot_discriminator_baselines(rows, cfg, output_root):
    """Plot already evaluated target-discriminator scores."""
    if not rows:
        return
    settings = baseline_get(cfg, "discriminator", {})
    plot_root = output_root / "plots" / "discriminator"; plot_root.mkdir(parents=True, exist_ok=True)
    for architecture in sorted({row["architecture"] for row in rows}):
        architecture_rows = [row for row in rows if row["architecture"] == architecture]
        architecture_root = plot_root / architecture
        architecture_root.mkdir(parents=True, exist_ok=True)
        for kind, filename in (("forecast", "lead_time_reverse_kl.png"), ("corruption", "corruption_strength_reverse_kl.png")):
            visual_corruption_scale = kind == "corruption"
            xlabel = ("Relative corruption severity (1 = corruption-specific maximum)"
                      if visual_corruption_scale else "Lead time (hours)")
            figure, axis = plt.subplots(figsize=(9, 5))
            targets = sorted({row["target"] for row in architecture_rows if row["kind"] == kind})
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(targets), 1)))
            for target_index, (color, target) in enumerate(zip(colors, targets)):
                series = sorted(
                    [row for row in architecture_rows if row["kind"] == kind and row["target"] == target and not row["is_era5_test_null"]],
                    key=lambda row: row["x"],
                )
                target_rows = [row for row in architecture_rows if row["kind"] == kind and row["target"] == target]
                x_values = (relative_corruption_coordinates(series, "x")
                            if visual_corruption_scale else [row["x"] for row in series])
                label = (corruption_range_label(target, series, "x")
                         if visual_corruption_scale else target)
                axis.errorbar(x_values, [row["score"] for row in series],
                              yerr=[row["stderr"] for row in series], marker=series_marker(target_index),
                              color=color, label=label)
                null = next(row for row in target_rows if row["is_era5_test_null"])
                axis.scatter([0], [null["score"]], marker="D", color=color, zorder=4)
            input_label = "four fields" if architecture.startswith("sfno_") else "T2M"
            comparison_label = "Lead Time" if kind == "forecast" else "Corruption Strength"
            axis.set(
                xlabel=xlabel, ylabel="Reverse-KL critic score",
                title=(f"Reverse-KL Critic vs {comparison_label}: {architecture} ({input_label})\n"
                       "Diamond at zero: ERA5 second-half null"),
            )
            if visual_corruption_scale:
                axis.set_xlim(0.0, 1.0)
            if settings.get("plot_yscale", "symlog") == "symlog":
                axis.set_yscale("symlog", linthresh=float(settings.get("plot_linthresh", 1e-2)))
            else:
                axis.set_yscale(str(settings.get("plot_yscale")))
            axis.grid(alpha=0.3, which="both")
            axis.legend()
            figure.tight_layout()
            save_figure_bundle(figure, architecture_root / filename, plot_type="discriminator_reverse_kl", dpi=220); plt.close(figure)


def evaluate_standard_metrics(cfg):
    """Evaluate non-learned baselines and persist all plot-ready statistics."""
    variables = variables_from_config(cfg)
    metric_names = metric_names_from_config(cfg)
    output_root = baseline_output_dir(cfg, variables)

    real_ds = select_level(safe_open_dataset(cfg.real_nc_file), cfg.get("level"))
    missing = [variable for variable in variables if variable not in real_ds.data_vars]
    if missing:
        raise ValueError(f"Variables missing from ERA5/reference data: {missing}")

    scratch_dir = baseline_get(cfg, "scratch_dir", None)
    if scratch_dir is not None:
        Path(str(scratch_dir)).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="baseline-full-statistics-",
        dir=None if scratch_dir is None else str(scratch_dir),
    ) as temporary_dir:
        model_train = select_level(
            select_era5_split(real_ds, cfg, "train", coverage="model"), cfg.get("level")
        )
        model_null = select_level(
            select_era5_split(real_ds, cfg, "null", coverage="model"), cfg.get("level")
        )
        max_samples = int(baseline_get(cfg, "eval_samples", cfg_get(cfg, "max_samples", 100)))
        train_indices = sample_time_indices(model_train, max_samples)
        null_indices = sample_time_indices(model_null, max_samples)
        null_stats = streaming_reference_stats(cfg, model_train, variables, train_indices)
        train_features = streaming_joint_features(
            cfg, model_train, variables, null_stats, train_indices, metric_names,
            Path(temporary_dir) / "model-train-null-scwd.dat",
            description="ERA5 days 1–15 model-range reference",
        )
        null_features = streaming_joint_features(
            cfg, model_null, variables, null_stats, null_indices, metric_names,
            Path(temporary_dir) / "model-second-half-null-scwd.dat",
            description="ERA5 days 16–end model-range null",
        )
        null_row = evaluate_era5_train_shift_metrics(
            null_features, train_features, variables, metric_names, cfg
        )
        null_row.update(label=ERA5_NULL_LABEL, is_null=True)
        null_scwd_diagnostics = []
        if "scwd" in metric_names:
            null_scwd_diagnostics.append(
                scwd_anchor_diagnostic(
                    null_features, train_features, cfg, ERA5_NULL_LABEL, 0,
                    null_row["scwd"], comparison_kind="null",
                )
            )
        lead_rows = [null_row]
        forecast_rows, scwd_diagnostics, global_mean_diagnostics = evaluate_lead_metrics(
            cfg, real_ds, variables, metric_names, temporary_dir,
        )
        lead_rows.extend(forecast_rows)
        corruption_reference_stats = normalization_stats_for_corruptions(
            cfg, real_ds, variables
        )
        corruption_rows, corruption_scwd_diagnostics = evaluate_corruption_metrics(
            cfg, real_ds, corruption_reference_stats, variables, metric_names, temporary_dir,
            return_scwd_diagnostics=True,
        )
        close_feature_memmaps(train_features)
        close_feature_memmaps(null_features)

    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=output_root / "resolved_config.yaml", resolve=True)
    lead_path = output_root / "data" / "lead_time.csv"
    corruption_path = output_root / "data" / "corruption_strength.csv"
    write_metric_csv(lead_rows, metric_names, lead_path, "lead_time")
    write_metric_csv(
        corruption_rows, metric_names, corruption_path, "corruption_strength"
    )
    write_scwd_anchor_diagnostics(
        null_scwd_diagnostics + scwd_diagnostics + corruption_scwd_diagnostics, output_root
    )
    write_global_mean_wasserstein_diagnostics(global_mean_diagnostics, variables, output_root)
    evaluate_corruption_disturbances(
        cfg, real_ds, corruption_reference_stats, variables, output_root
    )
    real_ds.close()
    paths = [output_root / "resolved_config.yaml", lead_path, corruption_path]
    paths += [
        path for path in (
            output_root / "data" / "scwd_anchor_contributions.nc",
            output_root / "data" / "global_mean_wasserstein_distributions.nc",
            output_root / "data" / "corruption_disturbances.nc",
        ) if path.is_file()
    ]
    return paths


def evaluate_discriminator_metrics(cfg):
    """Evaluate trained target critics independently of standard metrics."""
    variables = variables_from_config(cfg)
    output_root = baseline_output_dir(cfg, variables)
    real_ds = select_level(safe_open_dataset(cfg.real_nc_file), cfg.get("level"))
    missing = [variable for variable in variables if variable not in real_ds.data_vars]
    if missing:
        real_ds.close()
        raise ValueError(f"Variables missing from ERA5/reference data: {missing}")
    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=output_root / "resolved_config.yaml", resolve=True)
    discriminator_rows = discriminator_baseline_rows(cfg, real_ds, variables, output_root)
    write_discriminator_baselines(discriminator_rows, cfg, output_root)
    real_ds.close()
    data_path = output_root / "data" / "discriminator_reverse_kl.csv"
    return [output_root / "resolved_config.yaml"] + ([data_path] if data_path.is_file() else [])


def evaluate_standard_metric_baselines(cfg):
    """Compatibility wrapper evaluating both standard and learned baselines."""
    return evaluate_standard_metrics(cfg) + evaluate_discriminator_metrics(cfg)


def plot_saved_standard_metric_baselines(cfg):
    """Render baseline figures exclusively from persisted evaluation artifacts."""
    variables = variables_from_config(cfg)
    metric_names = metric_names_from_config(cfg)
    output_root = baseline_output_dir(cfg, variables)
    lead_rows = read_metric_csv(
        output_root / "data" / "lead_time.csv", metric_names, "lead_time"
    )
    corruption_rows = read_metric_csv(
        output_root / "data" / "corruption_strength.csv", metric_names, "corruption_strength"
    )
    scwd_diagnostics = read_scwd_anchor_diagnostics(output_root) if "scwd" in metric_names else []
    global_mean_diagnostics = (read_global_mean_wasserstein_diagnostics(output_root)
                               if "global_mean_wasserstein" in metric_names else [])
    discriminator_rows = read_discriminator_baselines(output_root)

    plot_lead_metrics(lead_rows, metric_names, variables, output_root)
    plot_corruption_metrics(corruption_rows, metric_names, variables, output_root)
    plot_normalized_lead_metrics_by_model(lead_rows, metric_names, variables, output_root)
    plot_normalized_corruption_metrics_by_type(corruption_rows, metric_names, variables, output_root)
    plot_corruption_disturbances(output_root)
    plot_scwd_anchor_diagnostics(scwd_diagnostics, output_root)
    plot_scwd_top_w1_distributions(scwd_diagnostics, cfg, output_root)
    plot_scwd_mean_response_differences(scwd_diagnostics, output_root)
    plot_global_mean_wasserstein_distributions(global_mean_diagnostics, output_root)
    plot_discriminator_baselines(discriminator_rows, cfg, output_root)
    mmd_root = mmd_global_moment_matching_output_dir(cfg, variables)
    plot_mmd_global_moment_matching(read_mmd_global_moment_matching(mmd_root), mmd_root)


@hydra.main(version_base=None, config_path="../conf", config_name="baseline_config")
def main(cfg: DictConfig):
    """Plot previously evaluated temporal-holdout-aligned baselines."""
    plot_saved_standard_metric_baselines(cfg)


if __name__ == "__main__":
    main()
