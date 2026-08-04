"""Visualize held-out forecast realism with temporal-discriminator logit maps."""

import csv
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from omegaconf import DictConfig
from tqdm import tqdm

try:
    from .analysis_utils import normalized_channels, normalization_stats, resolve_device
    from .temporal_holdout_utils import (
        discover_temporal_pairs,
        reconcile_checkpoint_variables,
        safe_model_name,
        temporal_checkpoint_dir,
        variables_from_config,
    )
    from .train_discriminator import WeatherDiscriminator, normalize_prediction_timedelta, safe_open_dataset, select_time_ranges
    from .monthly_split import concatenate_forecasts, forecast_pairs
except ImportError:
    from analysis_utils import normalized_channels, normalization_stats, resolve_device
    from temporal_holdout_utils import (
        discover_temporal_pairs,
        reconcile_checkpoint_variables,
        safe_model_name,
        temporal_checkpoint_dir,
        variables_from_config,
    )
    from train_discriminator import WeatherDiscriminator, normalize_prediction_timedelta, safe_open_dataset, select_time_ranges
    from monthly_split import concatenate_forecasts, forecast_pairs


def case_study_get(cfg, key, default):
    """Read an optional case-study setting from the main Hydra config."""
    value = cfg.get(key)
    return default if value is None else value


def lead_index(ds, lead_hour):
    """Return the forecast index for an integer lead hour."""
    ds = normalize_prediction_timedelta(ds)
    if "prediction_timedelta" not in ds.coords:
        if int(lead_hour) == 0:
            return None
        raise ValueError("Forecast dataset has no prediction_timedelta coordinate.")
    leads = np.asarray(ds.prediction_timedelta.values).astype(int)
    matches = np.flatnonzero(leads == int(lead_hour))
    if not matches.size:
        raise ValueError(f"Lead {lead_hour} h not found. Available leads: {leads.tolist()}")
    return int(matches[0])


def checkpoint_candidates(cfg, model_label):
    """Return candidate channel sets and temporal checkpoint paths in priority order."""
    requested_variables = case_study_get(cfg, "case_study_variables", None)
    candidates = [list(requested_variables)] if requested_variables else [variables_from_config(cfg)]
    selected = [str(cfg.selected_variable)]
    if selected not in candidates:
        candidates.append(selected)

    checkpoint_dir = temporal_checkpoint_dir(cfg)
    paths = []
    for variables in candidates:
        variable_tag = cfg.selected_variable if len(variables) == 1 else "all_fields"
        filename = (
            f"discriminator_{cfg.model_name}_{variable_tag}_"
            f"temporal_{safe_model_name(model_label)}.pth"
        )
        paths.append((variables, checkpoint_dir / filename))
    return paths


def resolve_checkpoint_and_variables(cfg, model_label):
    """Find a temporal checkpoint and the matching input-channel list."""
    for variables, path in checkpoint_candidates(cfg, model_label):
        if path.exists():
            return variables, path
    candidates = "\n".join(f"  {path}" for _, path in checkpoint_candidates(cfg, model_label))
    raise FileNotFoundError(f"No temporal checkpoint found for {model_label}. Tried:\n{candidates}")


def timestamp(value):
    """Normalize an xarray time coordinate to a pandas timestamp."""
    values = np.asarray(value)
    if values.ndim:
        if values.size != 1:
            raise ValueError(f"Expected one timestamp, received shape={values.shape}.")
        value = values.reshape(-1)[0]
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.Timestamp(str(value))


def select_initialization_date(forecast_ds, initialization_date):
    """Restrict a forecast dataset to one UTC initialization day when requested."""
    if initialization_date is None:
        return forecast_ds
    day_start = pd.Timestamp(str(initialization_date)).normalize()
    day_end = day_start + pd.Timedelta(days=1)
    times = np.asarray([timestamp(value) for value in forecast_ds.time.values])
    positions = np.flatnonzero((times >= day_start) & (times < day_end))
    if not positions.size:
        raise ValueError(
            f"No forecast initialization times on {day_start:%Y-%m-%d}; "
            f"available range is {times.min()} to {times.max()}."
        )
    return forecast_ds.isel(time=positions)


def matching_era5_index(era5_times, valid_time):
    """Return the exact ERA5 index; nearby timestamps are never substituted."""
    values = np.asarray(era5_times)
    if values.ndim != 1:
        values = values.reshape(-1)
    normalized = np.asarray(
        [timestamp(value).to_datetime64() for value in values],
        dtype="datetime64[ns]",
    )
    matches = np.flatnonzero(normalized == valid_time.to_datetime64())
    if len(matches) != 1:
        raise ValueError(f"Expected one exact ERA5 sample at {valid_time}, found {len(matches)}.")
    return int(matches[0]), 0.0


def field_values(ds_slice, variable):
    """Extract one field in latitude-longitude order for plotting."""
    return np.nan_to_num(ds_slice[variable].transpose("latitude", "longitude").values.astype(np.float32))


def score_forecast_samples(forecast_ds, variables, means, stds, lead_idx, model, device, batch_size):
    """Score every held-out forecast field at one lead time."""
    rows = []
    n_times = forecast_ds.sizes.get("time", 0)
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n_times, batch_size), desc="Scoring held-out forecasts"):
            stop = min(start + batch_size, n_times)
            inputs = []
            for time_idx in range(start, stop):
                sample = forecast_ds.isel(time=time_idx)
                if lead_idx is not None:
                    sample = sample.isel(prediction_timedelta=lead_idx)
                inputs.append(normalized_channels(sample, variables, means, stds))
            logits = model(torch.stack(inputs).to(device)).detach().cpu().numpy().reshape(-1)
            for time_idx, logit in zip(range(start, stop), logits):
                rows.append({"time_idx": time_idx, "logit": float(logit)})
    return rows


def select_logit_examples(rows, quantiles):
    """Choose distinct samples nearest requested empirical logit quantiles."""
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: row["logit"])
    selected = []
    used = set()
    for quantile in quantiles:
        target = float(np.quantile([row["logit"] for row in ordered], float(quantile)))
        choices = sorted(ordered, key=lambda row: (abs(row["logit"] - target), row["time_idx"]))
        choice = next((row for row in choices if row["time_idx"] not in used), None)
        if choice is not None:
            selected.append(dict(choice, quantile=float(quantile)))
            used.add(choice["time_idx"])
    return selected


def selected_payload(
    selected_rows,
    forecast_ds,
    era5_ds,
    era5_times,
    variables,
    plot_variable,
    means,
    stds,
    lead_idx,
    lead_hour,
    model,
    device,
):
    """Load aligned forecast/ERA5 fields and exact pre-pooling maps."""
    payload = []
    model.eval()
    with torch.no_grad():
        for row in selected_rows:
            forecast = forecast_ds.isel(time=row["time_idx"])
            input_slice = forecast if lead_idx is None else forecast.isel(prediction_timedelta=lead_idx)
            init_time = timestamp(forecast.time.values)
            valid_time = init_time + pd.to_timedelta(int(lead_hour), unit="h")
            era5_idx, alignment_hours = matching_era5_index(era5_times, valid_time)
            era5 = era5_ds.isel(time=era5_idx)

            inputs = normalized_channels(input_slice, variables, means, stds).unsqueeze(0).to(device)
            scalar_logit = model(inputs)
            mapped_logit, logit_map = model.forward_with_logit_map(inputs)
            if not torch.allclose(scalar_logit, mapped_logit, rtol=1e-5, atol=1e-5):
                raise RuntimeError("Spatial logit-map mean does not reproduce the discriminator logit.")

            forecast_field = field_values(input_slice, plot_variable)
            era5_field = field_values(era5, plot_variable)
            era5_inputs = normalized_channels(era5, variables, means, stds).unsqueeze(0).to(device)
            era5_logit, era5_logit_map = model.forward_with_logit_map(era5_inputs)
            if not torch.allclose(
                era5_logit, torch.mean(era5_logit_map, dim=(-2, -1)), rtol=1e-5, atol=1e-5
            ):
                raise RuntimeError("ERA5 spatial logit-map mean does not reproduce the discriminator logit.")
            map_values = logit_map[0, 0].detach().cpu().numpy()
            era5_map_values = era5_logit_map[0, 0].detach().cpu().numpy()
            upsampled_map = functional.interpolate(
                logit_map,
                size=forecast_field.shape,
                mode="nearest",
            )[0, 0].detach().cpu().numpy()
            upsampled_era5_map = functional.interpolate(
                era5_logit_map,
                size=era5_field.shape,
                mode="nearest",
            )[0, 0].detach().cpu().numpy()
            payload.append(
                {
                    **row,
                    "init_time": init_time,
                    "valid_time": valid_time,
                    "alignment_hours": alignment_hours,
                    "forecast": forecast_field,
                    "era5": era5_field,
                    "logit_map": map_values,
                    "upsampled_logit_map": upsampled_map,
                    "map_mean": float(np.mean(map_values)),
                    "era5_logit": float(era5_logit.item()),
                    "era5_logit_map": era5_map_values,
                    "upsampled_era5_logit_map": upsampled_era5_map,
                    "era5_map_mean": float(np.mean(era5_map_values)),
                }
            )
    return payload


def plot_case_study(payload, forecast_ds, model_label, plot_variable, lead_hour, output_path):
    """Plot forecast/ERA5 fields and their upsampled discriminator evidence maps."""
    latitudes = forecast_ds.latitude.values
    longitudes = forecast_ds.longitude.values
    fields = np.stack([item[key] for item in payload for key in ("forecast", "era5")])
    vmin, vmax = np.nanpercentile(fields, [1, 99])
    difference_limit = max(
        float(
            np.nanpercentile(
                np.abs(np.stack([item["forecast"] - item["era5"] for item in payload])), 99
            )
        ),
        1e-6,
    )
    logit_maps = np.stack(
        [
            logit_map
            for item in payload
            for logit_map in (item["upsampled_logit_map"], item["upsampled_era5_logit_map"])
        ]
    )
    map_limit = max(float(np.nanpercentile(np.abs(logit_maps), 99)), 1e-6)

    field_cmap = "inferno" if plot_variable in {"2m_temperature", "temperature"} else "coolwarm"
    figure, axes = plt.subplots(
        len(payload),
        5,
        figsize=(26, 4.3 * len(payload)),
        squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    for row_idx, item in enumerate(payload):
        title = (
            f"q={item['quantile']:.2f}, logit={item['logit']:.3f}, "
            f"init={item['init_time']:%Y-%m-%d %H:%M}, valid={item['valid_time']:%Y-%m-%d %H:%M}"
        )
        panels = [
            (f"{model_label} forecast +{lead_hour} h\n{title}", item["forecast"], field_cmap, vmin, vmax),
            ("ERA5 reference at valid time", item["era5"], field_cmap, vmin, vmax),
            (
                f"Field difference {model_label} - ERA5",
                item["forecast"] - item["era5"],
                "RdBu_r",
                -difference_limit,
                difference_limit,
            ),
            (
                f"{model_label} pre-pooling logit map (nearest-neighbour pixels)\n"
                f"mean={item['map_mean']:.3f}",
                item["upsampled_logit_map"],
                "RdBu_r",
                -map_limit,
                map_limit,
            ),
            (
                "ERA5 pre-pooling logit map (nearest-neighbour pixels)\n"
                f"mean={item['era5_map_mean']:.3f}",
                item["upsampled_era5_logit_map"],
                "RdBu_r",
                -map_limit,
                map_limit,
            ),
        ]
        for column_idx, (panel_title, values, cmap, panel_min, panel_max) in enumerate(panels):
            axis = axes[row_idx, column_idx]
            image = axis.pcolormesh(
                longitudes,
                latitudes,
                values,
                shading="auto",
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=panel_min,
                vmax=panel_max,
            )
            axis.set_global()
            axis.coastlines(linewidth=0.6)
            axis.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.5)
            axis.set_title(panel_title, fontsize=9)
            figure.colorbar(image, ax=axis, shrink=0.82)

    figure.suptitle(
        f"Temporal discriminator case study: {model_label}, {plot_variable}, +{lead_hour} h",
        fontsize=15,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_selected_samples(payload, output_path):
    """Write reproducible metadata for the plotted examples."""
    fieldnames = [
        "quantile",
        "time_idx",
        "init_time",
        "valid_time",
        "logit",
        "map_mean",
        "era5_logit",
        "era5_map_mean",
        "alignment_hours",
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in payload:
            writer.writerow({key: item[key] for key in fieldnames})


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Create a few interpretable temporal-holdout forecast examples."""
    model_label = str(case_study_get(cfg, "case_study_model", "GraphCast"))
    lead_hour = int(case_study_get(cfg, "case_study_lead_hour", 24))
    quantiles = list(case_study_get(cfg, "case_study_logit_quantiles", [0.05, 0.5, 0.95]))
    batch_size = int(case_study_get(cfg, "case_study_batch_size", cfg.batch_size))
    plot_variable = str(case_study_get(cfg, "case_study_plot_variable", cfg.selected_variable))
    initialization_date = case_study_get(cfg, "case_study_initialization_date", None)

    pairs = discover_temporal_pairs(cfg)
    if model_label not in pairs:
        available = ", ".join(sorted(pairs)) or "none"
        raise ValueError(f"No temporal test forecast file for {model_label}. Available: {available}")
    device = resolve_device()
    variables, checkpoint = resolve_checkpoint_and_variables(cfg, model_label)
    state_dict = torch.load(checkpoint, map_location=device)
    variables = reconcile_checkpoint_variables(cfg, variables, state_dict)
    if plot_variable not in variables:
        raise ValueError(f"case_study_plot_variable={plot_variable} is not one of {variables}")

    model = WeatherDiscriminator(
        len(variables), cfg.model_name, pretrained_backbone=False
    ).to(device)
    model.model.load_state_dict(state_dict)
    model.eval()

    forecast_ds = concatenate_forecasts([
        normalize_prediction_timedelta(safe_open_dataset(path))
        for path in pairs[model_label]["files"]
    ])
    missing = [variable for variable in variables if variable not in forecast_ds.data_vars]
    if missing:
        raise ValueError(f"Forecast file is missing model inputs: {missing}")
    era5_ds = safe_open_dataset(cfg.real_nc_file)
    if plot_variable not in era5_ds.data_vars:
        raise ValueError(f"ERA5 file is missing {plot_variable}")
    lead_idx = lead_index(forecast_ds, lead_hour)
    test_pairs = [
        pair for pair in forecast_pairs(forecast_ds, era5_ds, cfg, "test", [lead_hour])
        if pair.lead_index == lead_idx
    ]
    train_pairs = forecast_pairs(forecast_ds, era5_ds, cfg, "train", cfg.lead_times)
    forecast_ds = forecast_ds.isel(time=[pair.forecast_index for pair in test_pairs])
    forecast_ds = select_initialization_date(forecast_ds, initialization_date)
    era5_times = np.asarray(era5_ds.time.values)
    matched_train = era5_ds.isel(time=[pair.era5_index for pair in train_pairs])
    means = {variable: float(matched_train[variable].mean()) for variable in variables}
    stds = {variable: max(float(matched_train[variable].std()), 1e-8) for variable in variables}

    rows = score_forecast_samples(forecast_ds, variables, means, stds, lead_idx, model, device, batch_size)
    selected = select_logit_examples(rows, quantiles)
    if not selected:
        raise RuntimeError("No held-out forecast samples were scored.")
    payload = selected_payload(
        selected,
        forecast_ds,
        era5_ds,
        era5_times,
        variables,
        plot_variable,
        means,
        stds,
        lead_idx,
        lead_hour,
        model,
        device,
    )

    output_dir = Path(cfg.output_dir) / "temporal_logit_map_case_study"
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_model_name(model_label)
    date_tag = "" if initialization_date is None else f"_{pd.Timestamp(str(initialization_date)):%Y-%m-%d}"
    output_path = output_dir / f"{tag}_{plot_variable}_{lead_hour}h{date_tag}.png"
    metadata_path = output_dir / f"{tag}_{plot_variable}_{lead_hour}h{date_tag}_samples.csv"
    plot_case_study(payload, forecast_ds, model_label, plot_variable, lead_hour, output_path)
    write_selected_samples(payload, metadata_path)
    print(f"Saved case-study plot to: {output_path}")
    print(f"Saved selected-sample metadata to: {metadata_path}")

    forecast_ds.close()
    era5_ds.close()


if __name__ == "__main__":
    main()
