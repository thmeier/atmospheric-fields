"""Demonstrate a null space of the finite, sampled SCWD response operator.

This is deliberately implementation-specific: the perturbation is invisible
to the exact finite anchor matrix used by the baseline, not necessarily to the
continuous SCWD metric.
"""

import csv
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import hydra
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import lsmr
import xarray as xr
from omegaconf import DictConfig, OmegaConf

try:
    from .plot_standard_metric_baselines import (
        scwd_from_responses,
        scwd_sparse_weight_matrix_numpy,
        scwd_weight_vectors,
    )
    from .train_discriminator import safe_open_dataset, select_time_ranges
except ImportError:
    from plot_standard_metric_baselines import (
        scwd_from_responses,
        scwd_sparse_weight_matrix_numpy,
        scwd_weight_vectors,
    )
    from train_discriminator import safe_open_dataset, select_time_ranges


def experiment_get(cfg, key, default=None):
    section = cfg.get("scwd_null_space")
    if section is None:
        raise ValueError("Missing scwd_null_space configuration section.")
    value = section.get(key)
    return default if value is None else value


def cosine_weights(latitudes):
    return np.maximum(np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64))), 0.0)


def area_weighted_mean(values, latitudes):
    values = np.asarray(values, dtype=np.float64)
    weights = cosine_weights(latitudes)[:, None]
    return float(np.sum(values * weights) / (np.sum(weights) * values.shape[-1]))


def area_weighted_rms(values, latitudes):
    values = np.asarray(values, dtype=np.float64)
    weights = cosine_weights(latitudes)[:, None]
    return float(np.sqrt(np.sum(values**2 * weights) / (np.sum(weights) * values.shape[-1])))


def per_sample_area_weighted_rms(values, latitudes):
    """Return one cosine-area-weighted RMS for each leading sample."""
    values = np.asarray(values, dtype=np.float64)
    weights = cosine_weights(latitudes)[None, :, None]
    denominator = np.sum(weights) * values.shape[-1]
    return np.sqrt(np.sum(values**2 * weights, axis=(-2, -1)) / denominator)


def checkerboard_seed(shape, block_pixels=2):
    """Return a deterministic block checkerboard with no preferred location."""
    rows, columns = np.indices(tuple(int(value) for value in shape))
    block = max(1, int(block_pixels))
    return np.where(((rows // block) + (columns // block)) % 2, -1.0, 1.0)


def near_null_pattern(name, shape, latitudes):
    """Construct a human-visible pattern with predominantly sub-kernel scales."""
    rows, columns = np.indices(tuple(int(value) for value in shape))
    if name == "equatorial_checker_texture":
        carrier = np.where((rows + columns) % 2, -1.0, 1.0)
        latitude_envelope = 0.15 + 0.85 * np.exp(
            -(np.asarray(latitudes, dtype=np.float64)[:, None] / 25.0) ** 8
        )
        return carrier * latitude_envelope
    if name == "meridional_scanlines":
        return np.where(columns % 2, -1.0, 1.0)
    if name == "checkerboard_2px":
        return checkerboard_seed(shape, block_pixels=2)
    if name == "zonal_scanlines":
        return np.where(rows % 2, -1.0, 1.0)
    raise ValueError(f"Unknown near-null pattern: {name}.")


def pattern_label(name):
    return {
        "equatorial_checker_texture": "Equatorial checker texture",
        "meridional_scanlines": "Meridional scanlines",
        "checkerboard_2px": "2x2-pixel checkerboard",
        "zonal_scanlines": "Zonal scanlines",
    }.get(str(name), str(name).replace("_", " ").title())


def deranged_indices(size, seed=0):
    """Return a reproducible cyclic permutation with no self-pairs."""
    size = int(size)
    if size < 2:
        raise ValueError("Hemisphere splicing requires at least two ERA5 samples.")
    rng = np.random.default_rng(int(seed))
    shift = int(rng.integers(1, size))
    return np.roll(np.arange(size, dtype=int), shift)


def splice_hemispheres(northern, southern, latitudes, boundary_latitude=0.0):
    """Take the north from one field and south from another with a hard seam."""
    northern = np.asarray(northern)
    southern = np.asarray(southern)
    if northern.shape != southern.shape:
        raise ValueError("Northern and southern source fields must have identical shapes.")
    if northern.shape[-2] != len(latitudes):
        raise ValueError("Field latitude dimension does not match the latitude coordinate.")
    mask_shape = (1,) * (northern.ndim - 2) + (len(latitudes), 1)
    north_mask = (
        np.asarray(latitudes, dtype=np.float64) >= float(boundary_latitude)
    ).reshape(mask_shape)
    return np.where(north_mask, northern, southern)


def mean_constraint_row(latitudes, n_longitudes):
    row = np.repeat(cosine_weights(latitudes), int(n_longitudes)).astype(np.float64)
    norm = np.linalg.norm(row)
    if norm <= 0.0:
        raise ValueError("Latitude weights produced an empty mean constraint.")
    return sparse.csr_matrix((row / norm)[None, :])


def project_into_null_space(operator, seed, latitudes, atol=1e-10, btol=1e-10, maxiter=1000):
    """Orthogonally remove the response row space and weighted-mean mode."""
    seed = np.asarray(seed, dtype=np.float64)
    if operator.shape[1] != seed.size:
        raise ValueError(f"Operator has {operator.shape[1]} pixels but seed has {seed.size}.")
    augmented = sparse.vstack(
        [operator.astype(np.float64), mean_constraint_row(latitudes, seed.shape[-1])],
        format="csr",
    )
    solution = lsmr(
        augmented.T,
        seed.ravel(),
        atol=float(atol),
        btol=float(btol),
        maxiter=int(maxiter),
    )
    projected = np.asarray(seed.ravel() - augmented.T @ solution[0]).reshape(seed.shape)
    seed_norm = max(float(np.linalg.norm(seed.ravel())), 1e-15)
    response = np.asarray(augmented @ projected.ravel())
    diagnostics = {
        "solver_stop_code": int(solution[1]),
        "solver_iterations": int(solution[2]),
        "solver_residual_norm": float(solution[3]),
        "relative_response_residual": float(np.linalg.norm(response) / seed_norm),
        "maximum_absolute_response_residual": float(np.max(np.abs(response))),
        "retained_seed_norm_fraction": float(np.linalg.norm(projected.ravel()) / seed_norm),
    }
    return projected, diagnostics


def center_and_scale(values, latitudes, target_rms):
    centered = np.asarray(values, dtype=np.float64) - area_weighted_mean(values, latitudes)
    rms = area_weighted_rms(centered, latitudes)
    if rms <= 1e-15:
        raise ValueError("Cannot scale a zero-energy perturbation.")
    return centered * (float(target_rms) / rms)


def evenly_spaced_indices(size, maximum):
    if int(maximum) <= 0 or int(size) <= int(maximum):
        return np.arange(int(size), dtype=int)
    return np.linspace(0, int(size) - 1, int(maximum), dtype=int)


def training_statistics(dataset, variable, ranges):
    selected = select_time_ranges(dataset, ranges)
    if selected.sizes.get("time", 0) == 0:
        raise ValueError("No ERA5 training samples available for normalization.")
    mean = float(selected[variable].mean())
    std = float(selected[variable].std())
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError(f"Invalid ERA5 training standard deviation: {std}.")
    return mean, std


def normalized_fields(dataset, variable, indices, mean, std):
    fields = dataset[variable].isel(time=np.asarray(indices, dtype=int)).transpose(
        "time", "latitude", "longitude"
    ).values.astype(np.float64)
    return np.nan_to_num((fields - float(mean)) / float(std))


def filter_responses(fields, operator):
    fields = np.asarray(fields, dtype=np.float64)
    return np.asarray(fields.reshape(fields.shape[0], -1) @ operator.T, dtype=np.float64)


def scwd_from_response_arrays(candidate, reference, cfg):
    candidate = {"scwd_responses": np.asarray(candidate)[:, None, :]}
    reference = {"scwd_responses": np.asarray(reference)[:, None, :]}
    return scwd_from_responses(candidate, reference, cfg)


def local_response_w2(candidate, reference, n_quantiles=200):
    """Return one response-distribution W2 value per finite SCWD anchor."""
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("SCWD response arrays must have matching sample-by-anchor shapes.")
    quantiles = np.linspace(0.0, 1.0, int(n_quantiles))
    candidate_q = np.quantile(candidate, quantiles, axis=0)
    reference_q = np.quantile(reference, quantiles, axis=0)
    return np.sqrt(np.mean((candidate_q - reference_q) ** 2, axis=0))


def matching_time_index(dataset, requested):
    times = np.asarray(dataset.time.values).astype("datetime64[ns]")
    target = np.datetime64(str(requested), "ns")
    index = int(np.argmin(np.abs(times - target)))
    return index, times[index]


def write_metrics(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_map_features(axis, enabled):
    axis.set_global()
    if enabled:
        axis.coastlines(linewidth=0.5)
        axis.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)


def write_figure(era5_field, perturbed_fields, perturbations, rows, latitudes,
                 longitudes, time_value, output_path, coastlines=True):
    figure = plt.figure(figsize=(22, 9))
    grid = figure.add_gridspec(2, 4)
    all_fields = np.concatenate((era5_field.ravel(), perturbed_fields.ravel()))
    field_min, field_max = np.nanpercentile(all_fields, [1.0, 99.0])
    top_fields = [era5_field, *perturbed_fields]
    top_titles = [
        "ERA5",
        *[f"ERA5 + null perturbation\nRMS={row['physical_rms_k']:.2f} K" for row in rows],
    ]
    for column, (field, title) in enumerate(zip(top_fields, top_titles)):
        axis = figure.add_subplot(grid[0, column], projection=ccrs.PlateCarree())
        image = axis.pcolormesh(
            longitudes, latitudes, field, shading="auto", cmap="inferno",
            vmin=field_min, vmax=field_max, transform=ccrs.PlateCarree(),
        )
        add_map_features(axis, coastlines)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.75, label="2 m temperature (K)")

    metric_axis = figure.add_subplot(grid[1, 0])
    rms = np.asarray([row["normalized_rms"] for row in rows])
    null_scwd = np.asarray([row["null_space_scwd"] for row in rows])
    control_scwd = np.asarray([row["control_scwd"] for row in rows])
    metric_axis.plot(rms, null_scwd, marker="o", label="Projected null-space pattern")
    metric_axis.plot(rms, control_scwd, marker="o", label="Unprojected control")
    positive = np.concatenate((null_scwd[null_scwd > 0], control_scwd[control_scwd > 0]))
    linthresh = max(float(np.min(positive)) * 0.5, 1e-14) if positive.size else 1e-14
    metric_axis.set_yscale("symlog", linthresh=linthresh)
    metric_axis.set_xlabel("Perturbation RMS (ERA5-train standard deviations)")
    metric_axis.set_ylabel("SCWD to identical clean ERA5 samples")
    metric_axis.grid(True, alpha=0.3)
    metric_axis.legend(fontsize=8)
    metric_axis.set_title("Finite-operator blind spot")

    for column, (perturbation, row) in enumerate(zip(perturbations, rows), start=1):
        axis = figure.add_subplot(grid[1, column], projection=ccrs.PlateCarree())
        limit = max(float(np.nanmax(np.abs(perturbation))), 1e-12)
        image = axis.pcolormesh(
            longitudes, latitudes, perturbation, shading="auto", cmap="RdBu_r",
            vmin=-limit, vmax=limit, transform=ccrs.PlateCarree(),
        )
        add_map_features(axis, coastlines)
        axis.set_title(
            f"Signed perturbation\nRMS={row['physical_rms_k']:.2f} K, "
            f"SCWD={row['null_space_scwd']:.3g}"
        )
        figure.colorbar(image, ax=axis, shrink=0.75, label="Temperature difference (K)")
    figure.suptitle(
        f"SCWD finite-operator null-space case study | ERA5 "
        f"{np.datetime_as_string(time_value, unit='h')}\n"
        "Each residual panel uses its own symmetric colour scale",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_near_null_gallery(era5_field, perturbations, perturbed_fields, rows,
                            latitudes, longitudes, time_value, output_path,
                            coastlines=True):
    """Plot equally energetic structured perturbations and their ERA5 fields."""
    count = len(rows)
    if count == 0:
        return
    figure, axes = plt.subplots(
        2, count, figsize=(5.2 * count, 8.5), squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    perturbation_limit = max(float(np.nanmax(np.abs(perturbations))), 1e-12)
    all_fields = np.concatenate((era5_field.ravel(), perturbed_fields.ravel()))
    field_min, field_max = np.nanpercentile(all_fields, [1.0, 99.0])
    for column, row in enumerate(rows):
        residual_axis = axes[0, column]
        residual_image = residual_axis.pcolormesh(
            longitudes, latitudes, perturbations[column], shading="auto",
            cmap="RdBu_r", vmin=-perturbation_limit, vmax=perturbation_limit,
            transform=ccrs.PlateCarree(),
        )
        add_map_features(residual_axis, coastlines)
        residual_axis.set_title(
            f"{pattern_label(row['pattern'])}\n"
            f"SCWD={row['scwd']:.3g} | response RMS={row['unit_response_rms']:.3g}",
            fontsize=10,
        )
        field_axis = axes[1, column]
        field_image = field_axis.pcolormesh(
            longitudes, latitudes, perturbed_fields[column], shading="auto",
            cmap="inferno", vmin=field_min, vmax=field_max,
            transform=ccrs.PlateCarree(),
        )
        add_map_features(field_axis, coastlines)
        field_axis.set_title(
            f"Perturbed ERA5\nRMS={row['physical_rms_k']:.2f} K", fontsize=10
        )
    figure.colorbar(
        residual_image, ax=axes[0].tolist(), shrink=0.75,
        label="Temperature difference (K)",
    )
    figure.colorbar(
        field_image, ax=axes[1].tolist(), shrink=0.75,
        label="2 m temperature (K)",
    )
    figure.suptitle(
        f"Human-visible near-null SCWD perturbations | ERA5 "
        f"{np.datetime_as_string(time_value, unit='h')} | "
        f"normalized RMS={rows[0]['normalized_rms']:.2f}\n"
        "All perturbations share one residual colour scale",
        fontsize=14,
    )
    figure.subplots_adjust(left=0.03, right=0.91, bottom=0.06, top=0.88, wspace=0.08, hspace=0.12)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_hemisphere_splice_figure(north_source, south_source, spliced, difference,
                                   latitudes, longitudes, north_time, south_time,
                                   boundary, metrics, output_path, coastlines=True):
    """Show both source states, the hard splice, and its difference from source A."""
    figure, axes = plt.subplots(
        2, 2, figsize=(14, 8.5), squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    field_min, field_max = np.nanpercentile(
        np.concatenate((north_source.ravel(), south_source.ravel(), spliced.ravel())),
        [1.0, 99.0],
    )
    difference_limit = max(float(np.nanpercentile(np.abs(difference), 99.0)), 1e-12)
    panels = (
        ("Northern source", north_source, "inferno", field_min, field_max),
        ("Southern donor", south_source, "inferno", field_min, field_max),
        ("Hard hemisphere splice", spliced, "inferno", field_min, field_max),
        ("Splice - northern source", difference, "RdBu_r", -difference_limit, difference_limit),
    )
    for axis, (title, values, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        image = axis.pcolormesh(
            longitudes, latitudes, values, shading="auto", cmap=cmap,
            vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
        )
        add_map_features(axis, coastlines)
        axis.plot(
            [float(np.min(longitudes)), float(np.max(longitudes))],
            [float(boundary), float(boundary)], color="cyan", linewidth=0.8,
            transform=ccrs.PlateCarree(),
        )
        axis.set_title(title)
        figure.colorbar(
            image, ax=axis, shrink=0.76,
            label="Temperature difference (K)" if cmap == "RdBu_r" else "2 m temperature (K)",
        )
    figure.suptitle(
        "ERA5 hemisphere splice: locally valid marginals, globally inconsistent state\n"
        f"north={np.datetime_as_string(north_time, unit='h')} | "
        f"south={np.datetime_as_string(south_time, unit='h')} | "
        f"SCWD={metrics['scwd']:.4g} ({metrics['scwd_physical_k']:.3g} K) | "
        f"visual RMS={metrics['visual_rms_k']:.2f} K | "
        f"affected anchors={metrics['affected_anchor_fraction']:.1%}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_case_study(cfg):
    variable = str(experiment_get(cfg, "variable", "2m_temperature"))
    output_dir = Path(str(experiment_get(cfg, "output_dir")))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    era5 = safe_open_dataset(cfg.real_nc_file)
    if variable not in era5:
        raise ValueError(f"ERA5 does not contain configured variable {variable}.")
    train_mean, train_std = training_statistics(era5, variable, cfg.train_real_range)
    test = select_time_ranges(era5, cfg.test_real_ranges)
    if test.sizes.get("time", 0) == 0:
        raise ValueError("No ERA5 temporal-holdout samples are available.")

    latitudes = np.asarray(test.latitude.values, dtype=np.float64)
    longitudes = np.asarray(test.longitude.values, dtype=np.float64)
    weights = scwd_weight_vectors({"latitudes": latitudes, "longitudes": longitudes}, cfg)
    operator = scwd_sparse_weight_matrix_numpy(weights, len(latitudes) * len(longitudes))
    if operator is None or operator.shape[0] == 0:
        raise ValueError("SCWD produced no spatial anchors for the configured grid.")
    operator = operator.astype(np.float64)
    seed = checkerboard_seed(
        (len(latitudes), len(longitudes)), experiment_get(cfg, "seed_block_pixels", 2)
    )
    null_direction, diagnostics = project_into_null_space(
        operator, seed, latitudes,
        experiment_get(cfg, "projection_atol", 1e-10),
        experiment_get(cfg, "projection_btol", 1e-10),
        experiment_get(cfg, "projection_max_iterations", 1000),
    )
    tolerance = float(experiment_get(cfg, "verification_tolerance", 1e-7))
    if diagnostics["relative_response_residual"] > tolerance:
        raise RuntimeError(
            "Null-space projection did not converge to the requested tolerance: "
            f"{diagnostics['relative_response_residual']:.3g} > {tolerance:.3g}."
        )
    null_direction = center_and_scale(null_direction, latitudes, 1.0)
    control_direction = center_and_scale(seed, latitudes, 1.0)

    indices = evenly_spaced_indices(
        test.sizes["time"], experiment_get(cfg, "evaluation_samples", 128)
    )
    clean = normalized_fields(test, variable, indices, train_mean, train_std)
    clean_responses = filter_responses(clean, operator)
    clean_scwd = scwd_from_response_arrays(clean_responses, clean_responses, cfg)
    visual_index, visual_time = matching_time_index(
        test, experiment_get(cfg, "visualization_time", "2018-08-23T00:00:00")
    )
    visual = normalized_fields(test, variable, [visual_index], train_mean, train_std)[0]
    visual_physical = visual * train_std + train_mean

    rows, physical_perturbations, perturbed_fields = [], [], []
    for target_rms in experiment_get(cfg, "rms_values", [0.05, 0.10, 0.20]):
        target_rms = float(target_rms)
        null_perturbation = null_direction * target_rms
        control_perturbation = control_direction * target_rms
        null_responses = filter_responses(clean + null_perturbation[None], operator)
        control_responses = filter_responses(clean + control_perturbation[None], operator)
        physical = null_perturbation * train_std
        rows.append({
            "normalized_rms": target_rms,
            "physical_rms_k": area_weighted_rms(physical, latitudes),
            "area_weighted_mean_k": area_weighted_mean(physical, latitudes),
            "clean_scwd": clean_scwd,
            "null_space_scwd": scwd_from_response_arrays(null_responses, clean_responses, cfg),
            "control_scwd": scwd_from_response_arrays(control_responses, clean_responses, cfg),
            "evaluation_samples": len(indices),
            **diagnostics,
        })
        physical_perturbations.append(physical)
        perturbed_fields.append(visual_physical + physical)

    physical_perturbations = np.stack(physical_perturbations)
    perturbed_fields = np.stack(perturbed_fields)
    write_metrics(rows, output_dir / "metrics.csv")
    write_figure(
        visual_physical, perturbed_fields, physical_perturbations, rows,
        latitudes, longitudes, visual_time, output_dir / "case_study.png",
        bool(experiment_get(cfg, "plot_coastlines", True)),
    )

    gallery_rms = float(experiment_get(cfg, "near_null_gallery_rms", 0.20))
    gallery_names = list(experiment_get(cfg, "near_null_patterns", []))
    gallery_rows, gallery_perturbations, gallery_fields = [], [], []
    for name in gallery_names:
        raw_pattern = near_null_pattern(
            str(name), (len(latitudes), len(longitudes)), latitudes
        )
        direction = center_and_scale(raw_pattern, latitudes, 1.0)
        perturbation = direction * gallery_rms
        responses = filter_responses(clean + perturbation[None], operator)
        physical = perturbation * train_std
        gallery_rows.append({
            "pattern": str(name),
            "normalized_rms": gallery_rms,
            "physical_rms_k": area_weighted_rms(physical, latitudes),
            "area_weighted_mean_k": area_weighted_mean(physical, latitudes),
            "scwd": scwd_from_response_arrays(responses, clean_responses, cfg),
            "unit_response_rms": float(
                np.sqrt(np.mean(np.asarray(operator @ direction.ravel()) ** 2))
            ),
            "evaluation_samples": len(indices),
        })
        gallery_perturbations.append(physical)
        gallery_fields.append(visual_physical + physical)
    if gallery_rows:
        gallery_perturbations = np.stack(gallery_perturbations)
        gallery_fields = np.stack(gallery_fields)
        write_metrics(gallery_rows, output_dir / "near_null_metrics.csv")
        write_near_null_gallery(
            visual_physical, gallery_perturbations, gallery_fields, gallery_rows,
            latitudes, longitudes, visual_time,
            output_dir / "near_null_gallery.png",
            bool(experiment_get(cfg, "plot_coastlines", True)),
        )
        gallery_dataset = xr.Dataset(
            data_vars={
                "perturbation": (
                    ("pattern", "latitude", "longitude"),
                    gallery_perturbations.astype(np.float32),
                ),
                "perturbed_era5": (
                    ("pattern", "latitude", "longitude"),
                    gallery_fields.astype(np.float32),
                ),
                "scwd": (
                    "pattern", np.asarray([row["scwd"] for row in gallery_rows])
                ),
                "unit_response_rms": (
                    "pattern",
                    np.asarray([row["unit_response_rms"] for row in gallery_rows]),
                ),
            },
            coords={
                "pattern": gallery_names,
                "latitude": latitudes,
                "longitude": longitudes,
                "time": visual_time,
            },
            attrs={
                "description": "Structured near-null perturbations for the finite sampled SCWD operator",
                "normalized_rms": gallery_rms,
                "temperature_units": "K",
                "training_std_k": train_std,
            },
        )
        gallery_dataset.to_netcdf(output_dir / "near_null_gallery.nc")
        gallery_dataset.close()

    if bool(experiment_get(cfg, "hemisphere_splice_enabled", True)):
        splice_seed = int(experiment_get(cfg, "hemisphere_splice_seed", 0))
        splice_boundary = float(experiment_get(cfg, "hemisphere_splice_latitude", 0.0))
        donor_indices = deranged_indices(len(clean), splice_seed)
        spliced = splice_hemispheres(
            clean, clean[donor_indices], latitudes, splice_boundary
        )
        spliced_responses = filter_responses(spliced, operator)
        local_w2 = local_response_w2(
            spliced_responses,
            clean_responses,
            int(cfg.baseline.get("scwd_quantiles", 200)),
        )
        affected = local_w2 > 1e-8
        physical_differences = (spliced - clean) * train_std
        sample_rms_k = per_sample_area_weighted_rms(physical_differences, latitudes)

        visual_rng = np.random.default_rng(splice_seed)
        possible_donors = np.delete(np.arange(test.sizes["time"], dtype=int), visual_index)
        visual_donor_index = int(visual_rng.choice(possible_donors))
        visual_donor = normalized_fields(
            test, variable, [visual_donor_index], train_mean, train_std
        )[0]
        visual_donor_physical = visual_donor * train_std + train_mean
        visual_splice = splice_hemispheres(
            visual_physical, visual_donor_physical, latitudes, splice_boundary
        )
        visual_difference = visual_splice - visual_physical
        visual_donor_time = np.asarray(test.time.values).astype("datetime64[ns]")[visual_donor_index]
        splice_scwd = scwd_from_response_arrays(spliced_responses, clean_responses, cfg)
        splice_metrics = {
            "scwd": splice_scwd,
            "scwd_physical_k": splice_scwd * train_std,
            "evaluation_samples": len(indices),
            "donor_permutation_shift": int((donor_indices[0] - 0) % len(clean)),
            "splice_latitude": splice_boundary,
            "affected_anchor_count": int(np.sum(affected)),
            "total_anchor_count": int(len(local_w2)),
            "affected_anchor_fraction": float(np.mean(affected)),
            "maximum_local_w2_normalized": float(np.max(local_w2)),
            "maximum_local_w2_k": float(np.max(local_w2)) * train_std,
            "mean_rms_k": float(np.mean(sample_rms_k)),
            "median_rms_k": float(np.median(sample_rms_k)),
            "minimum_rms_k": float(np.min(sample_rms_k)),
            "maximum_rms_k": float(np.max(sample_rms_k)),
            "visual_rms_k": area_weighted_rms(visual_difference, latitudes),
            "northern_source_time": np.datetime_as_string(visual_time, unit="h"),
            "southern_donor_time": np.datetime_as_string(visual_donor_time, unit="h"),
        }
        write_metrics([splice_metrics], output_dir / "hemisphere_splice_metrics.csv")
        write_hemisphere_splice_figure(
            visual_physical, visual_donor_physical, visual_splice, visual_difference,
            latitudes, longitudes, visual_time, visual_donor_time, splice_boundary,
            splice_metrics, output_dir / "hemisphere_splice.png",
            bool(experiment_get(cfg, "plot_coastlines", True)),
        )
        splice_dataset = xr.Dataset(
            data_vars={
                "northern_source": (
                    ("latitude", "longitude"), visual_physical.astype(np.float32)
                ),
                "southern_donor": (
                    ("latitude", "longitude"), visual_donor_physical.astype(np.float32)
                ),
                "hemisphere_splice": (
                    ("latitude", "longitude"), visual_splice.astype(np.float32)
                ),
                "difference_from_northern_source": (
                    ("latitude", "longitude"), visual_difference.astype(np.float32)
                ),
            },
            coords={"latitude": latitudes, "longitude": longitudes},
            attrs={
                "description": "Northern ERA5 hemisphere joined to a randomly selected southern donor",
                "northern_source_time": splice_metrics["northern_source_time"],
                "southern_donor_time": splice_metrics["southern_donor_time"],
                "splice_latitude": splice_boundary,
                "scwd": splice_metrics["scwd"],
                "visual_rms_k": splice_metrics["visual_rms_k"],
                "temperature_units": "K",
            },
        )
        splice_dataset.to_netcdf(output_dir / "hemisphere_splice.nc")
        splice_dataset.close()
    dataset = xr.Dataset(
        data_vars={
            "era5": (("latitude", "longitude"), visual_physical.astype(np.float32)),
            "null_space_direction": (("latitude", "longitude"), null_direction.astype(np.float32)),
            "perturbation": (("rms_scale", "latitude", "longitude"), physical_perturbations.astype(np.float32)),
            "perturbed_era5": (("rms_scale", "latitude", "longitude"), perturbed_fields.astype(np.float32)),
        },
        coords={
            "rms_scale": np.asarray([row["normalized_rms"] for row in rows]),
            "latitude": latitudes,
            "longitude": longitudes,
            "time": visual_time,
        },
        attrs={
            "description": "Null space of the finite sampled SCWD operator, not continuous SCWD",
            "temperature_units": "K",
            "training_mean_k": train_mean,
            "training_std_k": train_std,
            **diagnostics,
        },
    )
    dataset.to_netcdf(output_dir / "case_study.nc")
    dataset.close()
    era5.close()
    print(f"Saved SCWD null-space case study to: {output_dir}")
    return rows


@hydra.main(version_base=None, config_path="../conf", config_name="scwd_null_space_config")
def main(cfg: DictConfig):
    run_case_study(cfg)


if __name__ == "__main__":
    main()
