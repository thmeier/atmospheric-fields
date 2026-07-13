"""Distributional spatial-realism baselines for forecast/corruption experiments.

The discriminator plots score whether samples look real, not whether individual
forecasts match paired ERA5 fields. This script therefore compares candidate
sample distributions against an ERA5 reference distribution. It deliberately
does not use per-sample forecast/ERA5 pairs. The configured variables are
standardized with ERA5 moments and evaluated jointly; configure a single
variable when field-wise metrics are desired.

Metrics:
- `mean_bias`: candidate mean minus ERA5 mean.
- `std_ratio_error`: candidate standard deviation divided by ERA5 standard
  deviation, minus one.
- `crps_like_field_energy`: half-energy distance between complete multi-field
  states, using cosine-latitude-weighted mean absolute differences.
- `zonal_energy_spectrum_l2`: relative L2 distance between mean zonal spectra.
- `sliced_wasserstein`: sliced Wasserstein distance between unweighted flattened
  spatial field distributions.
- `sliced_wasserstein_lon_corrected`: same metric after applying the spherical
  surface-Jacobian correction for lat-lon cell areas.
- `mmd_rbf`: maximum mean discrepancy between flattened spatial-field
  distributions, using a Gaussian RBF kernel with a median-distance bandwidth.
- `scwd`: spherical convolutional Wasserstein distance approximation. Scalar
  fields use the compact-Wendland quantile approximation from Garrett et al.
  (2024); multi-field states use random channel weights inside each spherical
  filter so every filter still maps a sample to one scalar response.
"""

import csv
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

try:
    from .train_discriminator import (
        apply_configured_corruption,
        normalize_prediction_timedelta,
        safe_open_dataset,
        select_time_ranges,
    )
except ImportError:
    from train_discriminator import (
        apply_configured_corruption,
        normalize_prediction_timedelta,
        safe_open_dataset,
        select_time_ranges,
    )


DEFAULT_METRICS = [
    "mean_bias",
    "std_ratio_error",
    "crps_like_field_energy",
    "zonal_energy_spectrum_l2",
    "sliced_wasserstein",
    "sliced_wasserstein_lon_corrected",
    "mmd_rbf",
    "scwd",
]

AVAILABLE_METRICS = {
    "mean_bias",
    "mean_abs_diff",
    "std_ratio_error",
    "std_abs_diff",
    "crps_like_field_energy",
    "zonal_energy_spectrum_l2",
    "sliced_wasserstein",
    "sliced_wasserstein_lon_corrected",
    "sliced_cramer_wold",
    "sliced_cramer_wold_lon_corrected",
    "mmd_rbf",
    "scwd",
}

_SCWD_WEIGHT_CACHE = {}
_FIELD_SELF_DISTANCE_CACHE = {}


def cfg_get(cfg, key, default):
    """Return a config value, treating explicit YAML null as missing."""
    value = cfg.get(key)
    return default if value is None else value


def variables_from_config(cfg):
    """Return variables for metric baselines."""
    variables = cfg_get(cfg, "standard_metric_variables", None)
    if variables:
        return list(variables)
    variables = cfg_get(cfg, "variables", None)
    if variables:
        return list(variables)
    return [cfg.selected_variable]


def metric_names_from_config(cfg):
    """Return requested distributional baseline metrics."""
    metric_names = list(cfg_get(cfg, "standard_metrics", DEFAULT_METRICS))
    unknown = sorted(set(metric_names) - AVAILABLE_METRICS)
    if unknown:
        raise ValueError(f"Unknown standard metric(s): {unknown}. Available metrics: {sorted(AVAILABLE_METRICS)}")
    return metric_names


def standard_real_file(cfg):
    """Return the ERA5/reference file for the standard metric baseline."""
    return cfg_get(cfg, "standard_metric_real_nc_file", cfg_get(cfg, "test_real_nc_file", cfg.real_nc_file))


def standard_comparison_files(cfg):
    """Return forecast files for the lead-time distribution baseline."""
    return cfg_get(cfg, "standard_metric_comparison_files", cfg.comparison_files)


def standard_fake_ranges(cfg):
    """Return forecast initialization-time ranges for distribution scoring."""
    return cfg_get(cfg, "standard_metric_test_fake_range", cfg.test_fake_range)


def standard_real_ranges(cfg):
    """Return ERA5/reference ranges for the reference distribution."""
    return cfg_get(cfg, "standard_metric_test_real_ranges", cfg.test_real_ranges)


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


def latitude_weights(latitudes):
    """Return cosine-latitude weights normalized by their mean."""
    weights = np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64)))
    return weights / np.nanmean(weights)


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
    if not bool(cfg_get(cfg, "standard_metric_filter_invalid_fields", True)):
        return False
    zero_atol = float(cfg_get(cfg, "standard_metric_invalid_zero_atol", 1e-12))
    min_std = float(cfg_get(cfg, "standard_metric_invalid_min_std", 1e-12))
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


def sliced_wasserstein_distance(candidate_vectors, reference_vectors, n_projections, seed):
    """Approximate Wasserstein distance with random 1D projections."""
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
    if not bool(cfg_get(cfg, "standard_metric_mmd_standardize", True)):
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


def mmd_rbf_bandwidth(candidate, reference, configured_bandwidth):
    """Choose an RBF bandwidth from config or the pooled median distance."""
    if configured_bandwidth is not None and float(configured_bandwidth) > 0:
        return float(configured_bandwidth)
    pooled = np.vstack([candidate, reference])
    distances = squared_euclidean_distances(pooled, pooled)
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
    bandwidth = mmd_rbf_bandwidth(candidate, reference, cfg_get(cfg, "standard_metric_mmd_bandwidth", None))
    k_xx = rbf_kernel_mean(candidate, candidate, bandwidth)
    k_yy = rbf_kernel_mean(reference, reference, bandwidth)
    k_xy = rbf_kernel_mean(candidate, reference, bandwidth)
    return float(np.sqrt(max(k_xx + k_yy - 2.0 * k_xy, 0.0)))


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

    max_values = int(cfg_get(cfg, "standard_metric_value_samples", 20000))
    values_per_sample_variable = max(1, max_values // max(len(common_keys) * len(variables), 1))
    per_variable_pixels = max(1, int(cfg_get(cfg, "standard_metric_swd_pixels", 4096)) // len(variables))

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
    n_lat = int(cfg_get(cfg, "standard_metric_scwd_anchor_lat_points", 60))
    n_lon = int(cfg_get(cfg, "standard_metric_scwd_anchor_lon_points", 120))
    domain_n_lat = int(cfg_get(cfg, "standard_metric_scwd_domain_lat_points", 361))
    domain_n_lon = int(cfg_get(cfg, "standard_metric_scwd_domain_lon_points", 720))
    radius_km = float(cfg_get(cfg, "standard_metric_scwd_radius_km", 1000.0))
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


def spherical_convolutional_wasserstein(candidate, reference, cfg):
    """Approximate SCWD using the quantile algorithm from Garrett et al. (2024)."""
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

    r = float(cfg_get(cfg, "standard_metric_scwd_order", 2.0))
    n_quantiles = int(cfg_get(cfg, "standard_metric_scwd_quantiles", 200))
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

    n_channel_projections = int(cfg_get(cfg, "standard_metric_scwd_channel_projections", 16))
    seed = int(cfg_get(cfg, "standard_metric_swd_seed", 0))
    rng = np.random.default_rng(seed)
    candidate_flat = candidate_fields.reshape(candidate_fields.shape[0], candidate_fields.shape[1], -1)
    reference_flat = reference_fields.reshape(reference_fields.shape[0], reference_fields.shape[1], -1)
    r = float(cfg_get(cfg, "standard_metric_scwd_order", 2.0))
    n_quantiles = int(cfg_get(cfg, "standard_metric_scwd_quantiles", 200))
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
    """Return E[d(Y,Y')] for the ERA5/reference field distribution."""
    fields = np.asarray(reference["fields"])
    latitudes = reference["latitudes"]
    if fields.size == 0 or latitudes is None:
        return np.nan
    chunk_size = int(cfg_get(cfg, "standard_metric_field_energy_chunk_size", 4))
    cache_key = (id(fields), tuple(np.round(np.asarray(latitudes, dtype=np.float64), 8)), chunk_size)
    if cache_key not in _FIELD_SELF_DISTANCE_CACHE:
        _FIELD_SELF_DISTANCE_CACHE[cache_key] = field_l1_pairwise_mean(fields, fields, latitudes, chunk_size)
    return _FIELD_SELF_DISTANCE_CACHE[cache_key]


def crps_like_field_energy(candidate, reference, cfg):
    """Half-energy distance with a weighted L1 distance between complete fields."""
    fields = np.asarray(candidate["fields"])
    reference_fields = np.asarray(reference["fields"])
    latitudes = reference["latitudes"]
    if fields.size == 0 or reference_fields.size == 0 or latitudes is None:
        return np.nan

    chunk_size = int(cfg_get(cfg, "standard_metric_field_energy_chunk_size", 4))
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
    vector_pixels = int(cfg_get(cfg, "standard_metric_swd_pixels", 4096))

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


def distribution_metric_values(candidate, reference, cfg):
    """Compute scalar distributional metrics from candidate/reference features."""
    candidate_spectrum = candidate["spectrum"]
    reference_spectrum = reference["spectrum"]
    common_spectrum_len = min(candidate_spectrum.size, reference_spectrum.size)
    if common_spectrum_len:
        spectrum_l2 = float(
            np.linalg.norm(candidate_spectrum[:common_spectrum_len] - reference_spectrum[:common_spectrum_len])
            / (np.linalg.norm(reference_spectrum[:common_spectrum_len]) + 1e-12)
        )
    else:
        spectrum_l2 = np.nan

    n_projections = int(cfg_get(cfg, "standard_metric_swd_projections", 64))
    seed = int(cfg_get(cfg, "standard_metric_swd_seed", 0))
    cramer_wold_bandwidth = cfg_get(cfg, "standard_metric_cramer_wold_bandwidth", None)
    candidate_unweighted = candidate.get("unweighted_vectors", candidate["vectors"])
    reference_unweighted = reference.get("unweighted_vectors", reference["vectors"])

    return {
        "mean_bias": candidate["mean"] - reference["mean"],
        "mean_abs_diff": abs(candidate["mean"] - reference["mean"]),
        "std_ratio_error": candidate["std"] / (reference["std"] + 1e-12) - 1.0,
        "std_abs_diff": abs(candidate["std"] - reference["std"]),
        "crps_like_field_energy": crps_like_field_energy(candidate, reference, cfg),
        "zonal_energy_spectrum_l2": spectrum_l2,
        "sliced_wasserstein": sliced_wasserstein_distance(
            candidate_unweighted, reference_unweighted, n_projections, seed
        ),
        "sliced_wasserstein_lon_corrected": sliced_wasserstein_distance(
            candidate["vectors"], reference["vectors"], n_projections, seed
        ),
        "sliced_cramer_wold": sliced_cramer_wold_distance(
            candidate_unweighted, reference_unweighted, n_projections, seed, cramer_wold_bandwidth
        ),
        "sliced_cramer_wold_lon_corrected": sliced_cramer_wold_distance(
            candidate["vectors"], reference["vectors"], n_projections, seed, cramer_wold_bandwidth
        ),
        "mmd_rbf": mmd_rbf_distance(candidate["vectors"], reference["vectors"], cfg),
        "scwd": spherical_convolutional_wasserstein(candidate, reference, cfg),
    }


def single_sample_feature(features, sample_idx):
    """Return feature dict for one candidate field."""
    sample_values = features["sample_values"][sample_idx]
    fields = features["fields"][sample_idx:sample_idx + 1]
    channel_fields = features["channel_fields"][sample_idx:sample_idx + 1] if "channel_fields" in features else None
    vectors = features["vectors"][sample_idx:sample_idx + 1]
    unweighted_vectors = features.get("unweighted_vectors", vectors)[sample_idx:sample_idx + 1]
    spectra = features["spectra"][sample_idx:sample_idx + 1]
    sample = {
        "values": sample_values,
        "sample_values": [sample_values],
        "spectrum": spectra[0] if spectra.size else np.array([], dtype=np.float64),
        "spectra": spectra,
        "fields": fields,
        "vectors": vectors,
        "unweighted_vectors": unweighted_vectors,
        "sample_keys": features.get("sample_keys", [])[sample_idx:sample_idx + 1],
        "latitudes": features["latitudes"],
        "longitudes": features["longitudes"],
        "mean_field": fields[0] if fields.size else np.array([], dtype=np.float64),
        "mean": float(np.nanmean(sample_values)) if sample_values.size else np.nan,
        "std": float(np.nanstd(sample_values)) if sample_values.size else np.nan,
    }
    if channel_fields is not None:
        sample["channel_fields"] = channel_fields
        sample["channel_mean_fields"] = channel_fields[0] if channel_fields.size else np.array([], dtype=np.float64)
    return sample


def selected_distribution_metric_values(candidate, reference, metric_names, cfg):
    """Compute selected full-distribution metrics."""
    metrics = distribution_metric_values(candidate, reference, cfg)
    metrics["n_samples"] = int(candidate.get("n_valid_samples", 0))
    return metrics


def joint_variable_name(variables):
    """Return the output variable label for the configured joint sample."""
    return "all_fields" if len(variables) > 1 else variables[0]


def reference_features_from_config(cfg, truth_ds, variables):
    """Build the ERA5 distribution used as ground-truth reference."""
    max_samples = int(cfg_get(cfg, "standard_metric_eval_samples", cfg_get(cfg, "max_samples", 100)))
    max_values = int(cfg_get(cfg, "standard_metric_value_samples", 20000))
    ref_ds = select_level(select_time_ranges(truth_ds, standard_real_ranges(cfg)), cfg.get("level"))
    time_indices = sample_time_indices(ref_ds, max_samples)
    ref_ds = ref_ds.isel(time=time_indices).load()
    time_indices = list(range(ref_ds.sizes.get("time", 0)))
    return distribution_features(cfg, ref_ds, variables, time_indices, max_values)


def evaluate_lead_metrics(cfg, reference_features, variables, metric_names):
    """Compare forecast lead-time distributions against the ERA5 distribution."""
    results = []
    max_samples = int(cfg_get(cfg, "standard_metric_eval_samples", cfg_get(cfg, "max_samples", 100)))
    max_values = int(cfg_get(cfg, "standard_metric_value_samples", 20000))
    reference_joint = joint_features_from_variables(reference_features, reference_features, variables, cfg)
    variable_name = joint_variable_name(variables)

    for label, path in standard_comparison_files(cfg).items():
        if not os.path.exists(path):
            print(f"Skipping {label}: file not found at {path}")
            continue

        forecast_ds = normalize_prediction_timedelta(safe_open_dataset(path))
        forecast_ds = select_level(select_time_ranges(forecast_ds, standard_fake_ranges(cfg)), cfg.get("level"))
        if forecast_ds.sizes.get("time", 0) == 0:
            print(f"Skipping {label}: no forecast samples in standard metric test fake ranges.")
            forecast_ds.close()
            continue

        missing = [variable for variable in variables if variable not in forecast_ds.data_vars]
        if missing:
            print(f"Skipping {label}: missing variables {missing}")
            forecast_ds.close()
            continue

        leads = lead_hours(forecast_ds)
        time_indices = sample_time_indices(forecast_ds, max_samples)
        forecast_ds = forecast_ds.isel(time=time_indices).load()
        time_indices = list(range(forecast_ds.sizes.get("time", 0)))

        for lead_idx, lead_hour in enumerate(tqdm(leads, desc=f"{label} lead distribution metrics")):
            candidate_features = distribution_features(
                cfg,
                forecast_ds,
                variables,
                time_indices,
                max_values,
                lead_idx=lead_idx,
            )
            base_row = {
                "experiment": "lead_time",
                "label": label,
                "variable": variable_name,
                "lead_hour": int(lead_hour),
                "corruption": "",
                "severity": np.nan,
            }
            candidate_joint = joint_features_from_variables(candidate_features, reference_features, variables, cfg)
            metrics = selected_distribution_metric_values(
                candidate_joint,
                reference_joint,
                metric_names,
                cfg,
            )
            row = dict(base_row)
            row.update({metric_name: metrics[metric_name] for metric_name in metric_names})
            row["n_samples"] = metrics["n_samples"]
            if row["n_samples"] == 0:
                print(f"Warning: no valid joint samples for {label} lead={lead_hour} variables={variables}")
            results.append(row)
        forecast_ds.close()

    return results


def corruption_types_from_config(cfg):
    """Return synthetic corruptions to include in metric baselines."""
    values = cfg_get(
        cfg,
        "standard_metric_corruptions",
        cfg_get(cfg, "corruption_kfold_types", cfg_get(cfg, "corruption_types", [])),
    )
    return [str(value) for value in values]


def corruption_levels(corruption_type, cfg):
    """Return severity levels for one train-time corruption."""
    n_steps = int(cfg_get(cfg, "standard_metric_corruption_steps", cfg_get(cfg, "corruption_kfold_plot_steps", 7)))
    max_severity = float(cfg_get(cfg, "standard_metric_corruption_max_severity", 2.0))
    if n_steps <= 1:
        return [max_severity]
    return np.linspace(0.0, max_severity, n_steps).tolist()


def normalization_stats_for_corruptions(cfg, truth_ds, variables):
    """Compute normalization stats required by training-time corruptions."""
    stats_ds = select_level(select_time_ranges(truth_ds, cfg.train_real_range), cfg.get("level"))
    means = {}
    stds = {}
    for variable in variables:
        means[variable] = float(stats_ds[variable].mean())
        std = float(stats_ds[variable].std())
        stds[variable] = std if std > 1e-8 else 1.0
    return means, stds


def evaluate_corruption_metrics(cfg, truth_ds, reference_features, variables, metric_names):
    """Compare corrupted ERA5 distributions against the ERA5 distribution."""
    results = []
    max_samples = int(cfg_get(cfg, "standard_metric_eval_samples", cfg_get(cfg, "max_samples", 100)))
    max_values = int(cfg_get(cfg, "standard_metric_value_samples", 20000))
    reference_joint = joint_features_from_variables(reference_features, reference_features, variables, cfg)
    variable_name = joint_variable_name(variables)
    eval_ds = select_level(select_time_ranges(truth_ds, standard_real_ranges(cfg)), cfg.get("level"))
    time_indices = sample_time_indices(eval_ds, max_samples)
    eval_ds = eval_ds.isel(time=time_indices).load()
    time_indices = list(range(eval_ds.sizes.get("time", 0)))
    means, stds = normalization_stats_for_corruptions(cfg, truth_ds, variables)

    for corruption_type in corruption_types_from_config(cfg):
        levels = corruption_levels(corruption_type, cfg)
        for severity in tqdm(levels, desc=f"{corruption_type} distribution metrics"):
            candidate_features = distribution_features(
                cfg,
                eval_ds,
                variables,
                time_indices,
                max_values,
                corruption_type=corruption_type,
                severity=severity,
                means=means,
                stds=stds,
            )
            base_row = {
                "experiment": "corruption_strength",
                "label": "ERA5 corrupted",
                "variable": variable_name,
                "lead_hour": np.nan,
                "corruption": corruption_type,
                "severity": float(severity),
            }
            candidate_joint = joint_features_from_variables(candidate_features, reference_features, variables, cfg)
            metrics = selected_distribution_metric_values(
                candidate_joint,
                reference_joint,
                metric_names,
                cfg,
            )
            row = dict(base_row)
            row.update({metric_name: metrics[metric_name] for metric_name in metric_names})
            row["n_samples"] = metrics["n_samples"]
            if row["n_samples"] == 0:
                print(
                    "Warning: no valid joint samples for "
                    f"corruption={corruption_type} severity={severity} variables={variables}"
                )
            results.append(row)

    return results


def write_csv(rows, metric_names, output_path):
    """Write metric rows to CSV."""
    fieldnames = (
        ["experiment", "label", "variable", "lead_hour", "corruption", "severity", "n_samples"]
        + metric_names
    )
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_lead_metrics(rows, metric_names, variables, cfg):
    """Plot metric-vs-lead curves."""
    lead_rows = [row for row in rows if row["experiment"] == "lead_time"]
    plot_variables = list(variables)
    if len(variables) > 1:
        plot_variables.append("all_fields")
    for variable in plot_variables:
        variable_rows = [row for row in lead_rows if row["variable"] == variable]
        if not variable_rows:
            continue

        fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)), squeeze=False)
        labels = sorted({row["label"] for row in variable_rows})
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))

        for metric_idx, metric_name in enumerate(metric_names):
            ax = axes[metric_idx, 0]
            for color, label in zip(colors, labels):
                series = sorted(
                    [row for row in variable_rows if row["label"] == label],
                    key=lambda row: row["lead_hour"],
                )
                x_values = [row["lead_hour"] for row in series]
                y_values = [row[metric_name] for row in series]
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.8,
                    color=color,
                    label=label,
                )
            ax.set_title(metric_name)
            ax.set_xlabel("Lead time (hours)")
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)

        axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
        fig.suptitle(f"Distributional Metrics vs Lead Time: {variable}", fontsize=15)
        fig.tight_layout(rect=[0, 0, 0.82, 0.96])
        output_path = Path(cfg.output_dir) / f"standard_distribution_metrics_vs_lead_time_{variable}.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved lead-time metric plot to: {output_path}")


def plot_corruption_metrics(rows, metric_names, variables, cfg):
    """Plot metric-vs-corruption-strength curves."""
    corruption_rows = [row for row in rows if row["experiment"] == "corruption_strength"]
    plot_variables = list(variables)
    if len(variables) > 1:
        plot_variables.append("all_fields")
    for variable in plot_variables:
        variable_rows = [row for row in corruption_rows if row["variable"] == variable]
        if not variable_rows:
            continue

        for metric_name in metric_names:
            corruptions = sorted({row["corruption"] for row in variable_rows})
            n_cols = 2
            n_rows = int(np.ceil(len(corruptions) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)
            axes = axes.flatten()

            for axis_idx, corruption_type in enumerate(corruptions):
                ax = axes[axis_idx]
                series = sorted(
                    [row for row in variable_rows if row["corruption"] == corruption_type],
                    key=lambda row: row["severity"],
                )
                x_values = [row["severity"] for row in series]
                y_values = [row[metric_name] for row in series]
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.8,
                )
                ax.set_title(corruption_type)
                ax.set_xlabel("Corruption severity")
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)

            for ax in axes[len(corruptions):]:
                ax.axis("off")

            fig.suptitle(f"{metric_name} vs Corruption Strength: {variable}", fontsize=15)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            output_path = Path(cfg.output_dir) / f"standard_distribution_{metric_name}_vs_corruption_strength_{variable}.png"
            fig.savefig(output_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved corruption metric plot to: {output_path}")

    if len(variables) > 1:
        plot_combined_corruption_metrics(corruption_rows, metric_names, cfg)
        plot_combined_corruption_metric_panel(corruption_rows, metric_names, cfg)


def plot_combined_corruption_metrics(corruption_rows, metric_names, cfg):
    """Plot all corruption families together for the all-fields aggregate."""
    variable_rows = [row for row in corruption_rows if row["variable"] == "all_fields"]
    if not variable_rows:
        return

    corruptions = sorted({row["corruption"] for row in variable_rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(corruptions), 1)))
    for metric_name in metric_names:
        fig, ax = plt.subplots(figsize=(11, 6))
        for color, corruption_type in zip(colors, corruptions):
            series = sorted(
                [row for row in variable_rows if row["corruption"] == corruption_type],
                key=lambda row: row["severity"],
            )
            x_values = [row["severity"] for row in series]
            y_values = [row[metric_name] for row in series]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.8,
                color=color,
                label=corruption_type,
            )

        ax.set_title(f"{metric_name} vs Corruption Strength: all_fields")
        ax.set_xlabel("Corruption severity")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        output_path = Path(cfg.output_dir) / f"standard_distribution_{metric_name}_vs_corruption_strength_all_fields_combined.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined all-fields corruption metric plot to: {output_path}")


def plot_combined_corruption_metric_panel(corruption_rows, metric_names, cfg):
    """Plot all all-fields corruption metrics in one lead-time-style panel."""
    variable_rows = [row for row in corruption_rows if row["variable"] == "all_fields"]
    if not variable_rows:
        return

    corruptions = sorted({row["corruption"] for row in variable_rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(corruptions), 1)))
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)), squeeze=False)

    for metric_idx, metric_name in enumerate(metric_names):
        ax = axes[metric_idx, 0]
        for color, corruption_type in zip(colors, corruptions):
            series = sorted(
                [row for row in variable_rows if row["corruption"] == corruption_type],
                key=lambda row: row["severity"],
            )
            x_values = [row["severity"] for row in series]
            y_values = [row[metric_name] for row in series]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.8,
                color=color,
                label=corruption_type,
            )
        ax.set_title(metric_name)
        ax.set_xlabel("Corruption severity")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.suptitle("Distributional Metrics vs Corruption Strength: all_fields", fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.82, 0.96])
    output_path = Path(cfg.output_dir) / "standard_distribution_metrics_vs_corruption_strength_all_fields.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined all-fields corruption metric panel to: {output_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="kfold_config")
def main(cfg: DictConfig):
    """Run distributional spatial-realism metric baselines."""
    variables = variables_from_config(cfg)
    metric_names = metric_names_from_config(cfg)

    real_ds = select_level(safe_open_dataset(standard_real_file(cfg)), cfg.get("level"))
    missing = [variable for variable in variables if variable not in real_ds.data_vars]
    if missing:
        raise ValueError(f"Variables missing from ERA5/reference data: {missing}")

    reference_features = reference_features_from_config(cfg, real_ds, variables)
    rows = []
    rows.extend(evaluate_lead_metrics(cfg, reference_features, variables, metric_names))
    rows.extend(evaluate_corruption_metrics(cfg, real_ds, reference_features, variables, metric_names))

    os.makedirs(cfg.output_dir, exist_ok=True)
    csv_path = Path(cfg.output_dir) / "standard_distribution_metric_baselines.csv"
    write_csv(rows, metric_names, csv_path)
    print(f"Saved standard distribution metric values to: {csv_path}")

    plot_lead_metrics(rows, metric_names, variables, cfg)
    plot_corruption_metrics(rows, metric_names, variables, cfg)
    real_ds.close()


if __name__ == "__main__":
    main()
