"""Visualize Keisler 850 hPa temperature beside matching ERA5 fields.

This is a diagnostic for unusually large Keisler temperature metric error bars.
By default it selects the five samples with the largest absolute spatial-mean
bias at the 12 h lead and plots Keisler, ERA5 at the forecast valid time, and
their difference.
"""

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

try:
    from .train_discriminator import normalize_prediction_timedelta, safe_open_dataset, select_time_ranges
except ImportError:
    from train_discriminator import normalize_prediction_timedelta, safe_open_dataset, select_time_ranges


def cfg_get(cfg, key, default):
    """Return a config value, treating explicit YAML null as missing."""
    value = cfg.get(key)
    return default if value is None else value


def select_level(ds, level):
    """Select a pressure level when the dataset has one."""
    if level is None:
        return ds
    if "level" in ds.dims:
        return ds.sel(level=level)
    if "pressure_level" in ds.dims:
        return ds.sel(pressure_level=level)
    return ds


def as_timestamp(value):
    """Convert xarray time coordinate values to pandas timestamps."""
    try:
        return pd.Timestamp(value)
    except Exception:
        return pd.Timestamp(str(value))


def valid_time(init_time, lead_hours):
    """Return forecast valid time from initialization time and lead hours."""
    return as_timestamp(init_time) + pd.to_timedelta(int(lead_hours), unit="h")


def lead_index(ds, lead_hour):
    """Return index of the requested forecast lead hour."""
    ds = normalize_prediction_timedelta(ds)
    if "prediction_timedelta" not in ds.coords:
        raise ValueError("Keisler dataset has no prediction_timedelta coordinate.")
    leads = np.asarray(ds.prediction_timedelta.values).astype(int)
    matches = np.where(leads == int(lead_hour))[0]
    if matches.size == 0:
        raise ValueError(f"Lead {lead_hour} h not found. Available leads: {leads.tolist()}")
    return int(matches[0])


def field_values(ds, variable):
    """Return one 2D field in latitude-longitude order."""
    values = ds[variable].transpose("latitude", "longitude").values.astype(np.float64)
    return np.nan_to_num(values)


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


def matching_era5_slice(era5_ds, era5_times, target_time):
    """Select matching ERA5 valid-time field, falling back to nearest with a warning."""
    deltas = np.asarray([abs((time - target_time).total_seconds()) for time in era5_times])
    match_idx = int(np.argmin(deltas))
    delta_hours = deltas[match_idx] / 3600.0
    if delta_hours > 0:
        print(
            f"Warning: no exact ERA5 match for {target_time}; "
            f"using nearest {era5_times[match_idx]} ({delta_hours:.1f} h away)."
        )
    return era5_ds.isel(time=match_idx)


def score_samples(keisler_ds, era5_ds, era5_times, variable, lead_idx, lead_hour, include_invalid):
    """Compute per-sample spatial-mean bias against matching ERA5 valid time."""
    rows = []
    skipped = {}
    for time_idx in range(keisler_ds.sizes.get("time", 0)):
        init_time = as_timestamp(keisler_ds.time.values[time_idx])
        target_time = valid_time(init_time, lead_hour)
        keisler_slice = keisler_ds.isel(time=time_idx, prediction_timedelta=lead_idx)
        era5_slice = matching_era5_slice(era5_ds, era5_times, target_time)
        keisler = field_values(keisler_slice, variable)
        era5 = field_values(era5_slice, variable)
        invalid_reason = invalid_field_reason(keisler)
        if invalid_reason is not None and not include_invalid:
            skipped[invalid_reason] = skipped.get(invalid_reason, 0) + 1
            continue
        if keisler.shape != era5.shape:
            print(f"Skipping time_idx={time_idx}: shape mismatch Keisler={keisler.shape}, ERA5={era5.shape}")
            continue
        bias = float(np.nanmean(keisler - era5))
        rows.append(
            {
                "time_idx": time_idx,
                "init_time": init_time,
                "valid_time": target_time,
                "mean_bias": bias,
                "abs_mean_bias": abs(bias),
                "invalid_reason": invalid_reason or "",
            }
        )
    if skipped:
        print("Skipped invalid Keisler fields:", skipped)
    return rows


def select_samples(scores, n_samples, mode):
    """Select samples for plotting."""
    if not scores:
        return []
    if mode == "first":
        return scores[:n_samples]
    if mode == "even":
        indices = np.linspace(0, len(scores) - 1, min(n_samples, len(scores)), dtype=int)
        return [scores[int(idx)] for idx in indices]
    if mode == "worst_mean_bias":
        return sorted(scores, key=lambda row: row["abs_mean_bias"], reverse=True)[:n_samples]
    raise ValueError(f"Unknown sample selection mode: {mode}")


def plot_samples(samples, keisler_ds, era5_ds, era5_times, variable, lead_idx, lead_hour, output_path):
    """Create side-by-side Keisler/ERA5/difference diagnostic plot."""
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.8 * n_rows), squeeze=False)
    lon = keisler_ds.longitude.values
    lat = keisler_ds.latitude.values

    all_fields = []
    all_diffs = []
    payload = []
    for sample in samples:
        keisler_slice = keisler_ds.isel(time=sample["time_idx"], prediction_timedelta=lead_idx)
        era5_slice = matching_era5_slice(era5_ds, era5_times, sample["valid_time"])
        keisler = field_values(keisler_slice, variable)
        era5 = field_values(era5_slice, variable)
        diff = keisler - era5
        all_fields.extend([keisler, era5])
        all_diffs.append(diff)
        payload.append((sample, keisler, era5, diff))

    vmin, vmax = np.nanpercentile(np.stack(all_fields), [1, 99])
    diff_abs = float(np.nanpercentile(np.abs(np.stack(all_diffs)), 99))
    diff_abs = max(diff_abs, 1e-6)

    for row_idx, (sample, keisler, era5, diff) in enumerate(payload):
        title_suffix = (
            f"init={sample['init_time']:%Y-%m-%d %H:%M}, "
            f"valid={sample['valid_time']:%Y-%m-%d %H:%M}, "
            f"mean diff={sample['mean_bias']:.2f} K"
        )
        panels = [
            ("Keisler 12h" if lead_hour == 12 else f"Keisler {lead_hour}h", keisler, "coolwarm", vmin, vmax),
            ("ERA5 valid time", era5, "coolwarm", vmin, vmax),
            ("Keisler - ERA5", diff, "RdBu_r", -diff_abs, diff_abs),
        ]
        for col_idx, (title, values, cmap, curr_vmin, curr_vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            mesh = ax.pcolormesh(lon, lat, values, shading="auto", cmap=cmap, vmin=curr_vmin, vmax=curr_vmax)
            ax.set_title(f"{title}\n{title_suffix}" if col_idx == 0 else title)
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
            fig.colorbar(mesh, ax=ax, shrink=0.8)

    fig.suptitle(f"Keisler {variable} vs ERA5 at {lead_hour} h lead", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="../conf", config_name="kfold_config")
def main(cfg: DictConfig):
    """Create Keisler temperature diagnostic visualizations."""
    variable = str(cfg_get(cfg, "visualize_variable", "temperature"))
    lead_hour = int(cfg_get(cfg, "visualize_lead_hour", 12))
    n_samples = int(cfg_get(cfg, "visualize_samples", 5))
    sample_mode = str(cfg_get(cfg, "visualize_sample_mode", "worst_mean_bias"))
    include_invalid = bool(cfg_get(cfg, "visualize_include_invalid", False))

    keisler_path = cfg.standard_metric_comparison_files["Keisler"]
    era5_path = cfg_get(cfg, "standard_metric_real_nc_file", cfg.real_nc_file)
    keisler_ds = normalize_prediction_timedelta(safe_open_dataset(keisler_path))
    era5_ds = safe_open_dataset(era5_path)

    keisler_ds = select_level(select_time_ranges(keisler_ds, cfg.standard_metric_test_fake_range), cfg.get("level"))
    era5_ds = select_level(era5_ds, cfg.get("level"))
    if variable not in keisler_ds.data_vars:
        raise ValueError(f"{variable} missing from Keisler file: {keisler_path}")
    if variable not in era5_ds.data_vars:
        raise ValueError(f"{variable} missing from ERA5 file: {era5_path}")

    lead_idx = lead_index(keisler_ds, lead_hour)
    era5_times = np.asarray([as_timestamp(value) for value in era5_ds.time.values])
    scores = score_samples(keisler_ds, era5_ds, era5_times, variable, lead_idx, lead_hour, include_invalid)
    samples = select_samples(scores, n_samples, sample_mode)
    if not samples:
        raise RuntimeError("No Keisler samples were selected for visualization.")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"keisler_{variable}_{lead_hour}h_vs_era5_{sample_mode}_{len(samples)}samples.png"
    plot_samples(samples, keisler_ds, era5_ds, era5_times, variable, lead_idx, lead_hour, output_path)
    print(f"Saved Keisler diagnostic plot to: {output_path}")
    print("Selected samples:")
    for sample in samples:
        print(
            f"  idx={sample['time_idx']} init={sample['init_time']} "
            f"valid={sample['valid_time']} mean_bias={sample['mean_bias']:.3f} K "
            f"invalid={sample['invalid_reason'] or 'no'}"
        )
    keisler_ds.close()
    era5_ds.close()


if __name__ == "__main__":
    main()
