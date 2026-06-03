"""Poster figure: variational reverse-KL score for corruptions and forecasts.

The discriminator is trained with BCE, so its raw logit v(x) approximates
log p_X(x) / p_Xi(x) for the training reference Xi. For reverse KL in the
f-divergence convention D_f(X || Q), f(u) = -log(u), the optimal variational
critic is T(x) = f'(p_X / p_Xi) = -exp(-v(x)) and
f*(T(x)) = -1 - log(-T(x)) = v(x) - 1.

For any candidate forecast/corruption distribution Q, this script plots
S(Q) = E_X[T(x)] - E_Q[f*(T(y))].
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from corruptions import apply_high_freq_noise, apply_wind_channel_rotation
from train_discriminator import (
    WeatherDiscriminator,
    safe_open_dataset,
    validate_no_train_test_overlap,
    variables_from_config,
)


INK_BLACK = "#0D1821"
YALE_BLUE = "#344966"
BLUSH_PINK = "#E6AACE"
PORCELAIN = "#F0F4EF"
AMBER = "#C28F2C"
FORECAST_COLORS = {"IFS HRES": YALE_BLUE, "GraphCast": BLUSH_PINK, "Pangu-Weather": AMBER, "Pangu": AMBER}
FORECAST_MARKERS = {"IFS HRES": "^", "GraphCast": "D", "Pangu-Weather": "v", "Pangu": "v"}
FORECAST_DISPLAY_LABELS = {"IFS HRES": "IFS HRES (numerical)"}

mpl.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


DEFAULT_DATA_DIR = Path("/cluster/courses/pmlr/teams/team07/data")
DEFAULT_REAL = DEFAULT_DATA_DIR / "era5-gt_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc"
DEFAULT_PANGU_FORECAST = DEFAULT_DATA_DIR / "pangu_6steps_surf_1.5deg_2018-01-01_2018-12-31.nc"
SCORE_KIND = "reverse_kl_variational_s_q"
EXP_CLAMP_MAX = 50.0


class ScoreDataset(Dataset):
    def __init__(
        self,
        ds,
        variables,
        means,
        stds,
        max_samples=0,
        lead_index=None,
        transform_fn=None,
    ):
        self.ds = ds
        self.variables = list(variables)
        self.means = means
        self.stds = stds
        self.lead_index = lead_index
        self.transform_fn = transform_fn

        n_time = len(self.ds.time)
        if max_samples and n_time > max_samples:
            self.indices = np.linspace(0, n_time - 1, max_samples, dtype=int)
        else:
            self.indices = np.arange(n_time, dtype=int)

        missing = [v for v in self.variables if v not in self.ds.data_vars]
        if missing:
            raise ValueError(f"Missing variables in dataset: {missing}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ds_slice = self.ds.isel(time=int(self.indices[idx]))
        if self.lead_index is not None:
            ds_slice = ds_slice.isel(prediction_timedelta=self.lead_index)

        channels = []
        for variable in self.variables:
            raw = ds_slice[variable].values.astype(np.float32)
            std = self.stds[variable] if self.stds[variable] > 1e-8 else 1.0
            normalized = (raw - self.means[variable]) / std
            channels.append(np.nan_to_num(normalized, nan=0.0))

        sample = torch.tensor(np.stack(channels), dtype=torch.float32)
        if self.transform_fn is not None:
            sample = self.transform_fn(sample.unsqueeze(0)).squeeze(0)
        return sample


def to_int_lead_hours(values):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[h]").astype(int)
    return values.astype(int)


def select_time_range(ds, time_range):
    if not time_range:
        return ds
    return ds.sel(time=slice(time_range[0], time_range[1]))


def select_time_ranges(ds, time_ranges):
    if not time_ranges:
        return ds

    if len(time_ranges) == 2 and all(t is None or isinstance(t, str) for t in time_ranges):
        return select_time_range(ds, time_ranges)

    selected = []
    for time_range in time_ranges:
        part = select_time_range(ds, time_range)
        if "time" not in part.dims or len(part.time) > 0:
            selected.append(part)

    if not selected:
        raise ValueError(f"No ERA5 samples found in configured time ranges={time_ranges}")
    if len(selected) == 1:
        return selected[0]
    return xr.concat(selected, dim="time").sortby("time")


def select_zero_lead(ds):
    if "prediction_timedelta" not in ds.dims:
        return ds, None
    lead_hours = to_int_lead_hours(ds.prediction_timedelta.values)
    zero = np.where(lead_hours == 0)[0]
    return ds, int(zero[0]) if len(zero) else 0


def stats_from_real(real_ds, variables, train_range, time_chunk=32):
    ranges = [train_range] if (
        len(train_range) == 2 and all(t is None or isinstance(t, str) for t in train_range)
    ) else train_range

    means = {}
    stds = {}
    for variable in variables:
        total = 0.0
        total_sq = 0.0
        count = 0
        for curr_range in ranges:
            curr_ds = select_time_range(real_ds, curr_range)
            n_time = curr_ds.sizes.get("time", 1)
            for start in range(0, n_time, time_chunk):
                chunk = curr_ds[variable].isel(time=slice(start, start + time_chunk)).values
                chunk = np.asarray(chunk, dtype=np.float64)
                finite = np.isfinite(chunk)
                if not finite.any():
                    continue
                values = chunk[finite]
                total += float(values.sum())
                total_sq += float(np.square(values).sum())
                count += int(values.size)

        if count == 0:
            raise ValueError(f"No finite values found while computing stats for {variable}")
        mean = total / count
        variance = max(total_sq / count - mean ** 2, 0.0)
        means[variable] = float(mean)
        stds[variable] = float(np.sqrt(variance))

    return means, stds


def expectation_over_dataset(model, dataset, batch_size, num_workers, device, transform, desc):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    values = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device)).flatten()
            values.append(transform(logits).detach().cpu().numpy())
    if not values:
        raise ValueError(f"No samples available for {desc}")
    values = np.concatenate(values)
    mean = float(values.mean())
    stderr = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    print(f"{desc}: n={len(values)} mean={mean:.4f} stderr={stderr:.4f}")
    return mean, stderr


def reverse_kl_critic_from_logits(logits):
    return -torch.exp((-logits).clamp(max=EXP_CLAMP_MAX))


def reverse_kl_conjugate_from_logits(logits):
    return logits - 1.0


def reverse_kl_score(model, dataset, real_term, real_term_stderr, batch_size, num_workers, device, desc):
    conjugate_mean, conjugate_stderr = expectation_over_dataset(
        model,
        dataset,
        batch_size,
        num_workers,
        device,
        reverse_kl_conjugate_from_logits,
        f"{desc} f*(T)",
    )
    score = real_term - conjugate_mean
    stderr = float(np.sqrt(real_term_stderr ** 2 + conjugate_stderr ** 2))
    print(
        f"{desc}: S(Q)={score:.4f} stderr={stderr:.4f} "
        f"(E_X[T]={real_term:.4f}, E_Q[f*(T)]={conjugate_mean:.4f})"
    )
    return score, stderr


def load_model(cfg, variables, device, model_path=None):
    model = WeatherDiscriminator(len(variables), cfg.model_name).to(device)
    variable_tag = cfg.get("selected_variable", "all_fields") if len(variables) == 1 else "all_fields"
    if model_path is None:
        model_path = Path(cfg.output_dir) / f"weather_discriminator_{cfg.model_name}_{variable_tag}_lightning.pth"
    else:
        model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    model.model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, model_path


def compute_real_term(model, real_ds, variables, means, stds, real_lead_index, args, device, desc):
    real_dataset = ScoreDataset(
        real_ds, variables, means, stds, max_samples=args.n_samples, lead_index=real_lead_index
    )
    return expectation_over_dataset(
        model,
        real_dataset,
        args.batch_size,
        args.num_workers,
        device,
        reverse_kl_critic_from_logits,
        desc,
    )


def compute_forecast_curve(label, path, model, real_term, real_term_stderr, variables, means, stds, args, cfg, device):
    path = Path(path)
    if not path.exists():
        print(f"Skipping {label}: file not found at {path}")
        return None
    forecast_ds = select_time_range(safe_open_dataset(path), cfg.test_fake_range)
    if "prediction_timedelta" not in forecast_ds.dims:
        print(f"Skipping {label}: no prediction_timedelta dimension")
        return None

    lead_hours = to_int_lead_hours(forecast_ds.prediction_timedelta.values)
    keep = [i for i, h in enumerate(lead_hours) if int(h) in set(cfg.lead_times)]
    x_leads = []
    y_score = []
    y_stderr = []
    for lead_index in keep:
        lead = int(lead_hours[lead_index])
        dataset = ScoreDataset(
            forecast_ds, variables, means, stds,
            max_samples=args.n_samples,
            lead_index=lead_index,
        )
        mean, stderr = reverse_kl_score(
            model, dataset, real_term, real_term_stderr,
            args.batch_size, args.num_workers, device, f"{label} {lead}h"
        )
        x_leads.append(lead)
        y_score.append(mean)
        y_stderr.append(stderr)

    order = np.argsort(x_leads)
    return (
        np.asarray(x_leads)[order],
        np.asarray(y_score)[order],
        np.asarray(y_stderr)[order],
    )


def compute_curves(args, cfg):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variables = variables_from_config(cfg)
    real_path = Path(args.real_file or cfg.get("test_real_nc_file", DEFAULT_REAL))
    comparison_files = cfg.get("comparison_files", {})
    ifs_path = Path(args.ifs_file or comparison_files["IFS HRES"])
    graphcast_path = Path(args.graphcast_file or comparison_files["GraphCast"])
    pangu_data_dir = Path(cfg.get("data_dir", DEFAULT_DATA_DIR))
    pangu_path = Path(args.pangu_file or (pangu_data_dir / DEFAULT_PANGU_FORECAST.name))
    if args.pangu_holdout_model is None:
        variable_tag = cfg.get("selected_variable", "all_fields") if len(variables) == 1 else "all_fields"
        pangu_model_path = Path(cfg.output_dir) / f"weather_discriminator_{cfg.model_name}_{variable_tag}_pangu_holdout_lightning.pth"
    else:
        pangu_model_path = Path(args.pangu_holdout_model)

    real_ds = select_time_ranges(safe_open_dataset(real_path), cfg.test_real_ranges)
    real_ds, real_lead_index = select_zero_lead(real_ds)
    means, stds = stats_from_real(safe_open_dataset(real_path), variables, cfg.train_real_range)
    model, model_path = load_model(cfg, variables, device)

    print(f"Device: {device}")
    print(f"Model:  {model_path}")
    print(f"Fields: {variables}")

    real_term, real_term_stderr = compute_real_term(
        model,
        real_ds,
        variables,
        means,
        stds,
        real_lead_index,
        args,
        device,
        "ERA5 clean reference T",
    )

    severities = np.linspace(0.0, 1.0, args.n_severity_steps)
    hf_noise_score = []
    hf_noise_stderr = []
    wind_rotation_score = []
    wind_rotation_stderr = []
    for severity in severities:
        hf_dataset = ScoreDataset(
            real_ds, variables, means, stds,
            max_samples=args.n_samples,
            lead_index=real_lead_index,
            transform_fn=lambda x, s=float(severity): apply_high_freq_noise(x, s),
        )
        rot_dataset = ScoreDataset(
            real_ds, variables, means, stds,
            max_samples=args.n_samples,
            lead_index=real_lead_index,
            transform_fn=lambda x, s=float(severity): apply_wind_channel_rotation(x, s),
        )
        hf_mean, hf_se = reverse_kl_score(
            model, hf_dataset, real_term, real_term_stderr,
            args.batch_size, args.num_workers, device, f"HF noise {severity:.2f}"
        )
        rot_mean, rot_se = reverse_kl_score(
            model, rot_dataset, real_term, real_term_stderr,
            args.batch_size, args.num_workers, device, f"Wind rotation {severity:.2f}"
        )
        hf_noise_score.append(hf_mean)
        hf_noise_stderr.append(hf_se)
        wind_rotation_score.append(rot_mean)
        wind_rotation_stderr.append(rot_se)

    forecast_results = {}
    for label, path in {"IFS HRES": ifs_path, "GraphCast": graphcast_path}.items():
        curve = compute_forecast_curve(
            label, path, model, real_term, real_term_stderr,
            variables, means, stds, args, cfg, device,
        )
        if curve is not None:
            forecast_results[label] = curve

    if pangu_model_path.exists():
        pangu_model, loaded_pangu_model_path = load_model(cfg, variables, device, pangu_model_path)
        print(f"Pangu holdout model: {loaded_pangu_model_path}")
        pangu_real_term, pangu_real_term_stderr = compute_real_term(
            pangu_model,
            real_ds,
            variables,
            means,
            stds,
            real_lead_index,
            args,
            device,
            "ERA5 clean reference T (Pangu holdout model)",
        )
        pangu_curve = compute_forecast_curve(
            "Pangu-Weather", pangu_path, pangu_model, pangu_real_term, pangu_real_term_stderr,
            variables, means, stds, args, cfg, device,
        )
        if pangu_curve is not None:
            forecast_results["Pangu-Weather"] = pangu_curve
    else:
        print(f"Skipping Pangu-Weather S(Q): Pangu holdout model not found at {pangu_model_path}")

    return {
        "severities": severities,
        "hf_noise_score": np.asarray(hf_noise_score),
        "hf_noise_stderr": np.asarray(hf_noise_stderr),
        "wind_rotation_score": np.asarray(wind_rotation_score),
        "wind_rotation_stderr": np.asarray(wind_rotation_stderr),
        "forecast_results": forecast_results,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "model_path": str(model_path),
        "score_kind": SCORE_KIND,
        "test_real_ranges": str(cfg.test_real_ranges),
        "real_file": str(real_path),
        "real_term": real_term,
        "real_term_stderr": real_term_stderr,
        "pangu_model_path": str(pangu_model_path),
        "pangu_file": str(pangu_path),
    }


def save_cache(cache_path, curves):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "severities": curves["severities"],
        "hf_noise_score": curves["hf_noise_score"],
        "hf_noise_stderr": curves["hf_noise_stderr"],
        "wind_rotation_score": curves["wind_rotation_score"],
        "wind_rotation_stderr": curves["wind_rotation_stderr"],
        "n_samples": np.array(curves["n_samples"]),
        "seed": np.array(curves["seed"]),
        "model_path": np.array(curves["model_path"]),
        "score_kind": np.array(curves["score_kind"]),
        "test_real_ranges": np.array(curves["test_real_ranges"]),
        "real_file": np.array(curves["real_file"]),
        "real_term": np.array(curves["real_term"]),
        "real_term_stderr": np.array(curves["real_term_stderr"]),
        "pangu_model_path": np.array(curves.get("pangu_model_path", "")),
        "pangu_file": np.array(curves.get("pangu_file", "")),
    }
    for label, (leads, values, stderr) in curves["forecast_results"].items():
        key = label.lower().replace(" ", "_")
        payload[f"{key}_leads"] = leads
        payload[f"{key}_score"] = values
        payload[f"{key}_stderr"] = stderr
    np.savez(cache_path, **payload)
    print(f"Cached curves -> {cache_path}")


def load_cache(cache_path, expected_test_real_ranges=None, expected_real_file=None):
    data = np.load(cache_path, allow_pickle=True)
    score_kind = str(data["score_kind"]) if "score_kind" in data else ""
    if score_kind != SCORE_KIND:
        print(f"Ignoring stale cache with score_kind='{score_kind or 'legacy_neg_logit'}' <- {cache_path}")
        return None
    if "test_real_ranges" not in data:
        print(f"Ignoring stale cache without test_real_ranges metadata <- {cache_path}")
        return None
    cached_test_real_ranges = str(data["test_real_ranges"])
    if expected_test_real_ranges is not None and cached_test_real_ranges != str(expected_test_real_ranges):
        print(f"Ignoring stale cache with test_real_ranges={cached_test_real_ranges} <- {cache_path}")
        return None
    cached_real_file = str(data["real_file"]) if "real_file" in data else ""
    if expected_real_file is not None and cached_real_file != str(expected_real_file):
        print(f"Ignoring stale cache with real_file='{cached_real_file or 'unknown'}' <- {cache_path}")
        return None
    forecast_results = {}
    for label in ("IFS HRES", "GraphCast", "Pangu-Weather"):
        key = label.lower().replace(" ", "_")
        lead_key = f"{key}_leads"
        score_key = f"{key}_score"
        stderr_key = f"{key}_stderr"
        if lead_key in data and score_key in data:
            stderr = data[stderr_key] if stderr_key in data else np.zeros_like(data[score_key])
            forecast_results[label] = (data[lead_key], data[score_key], stderr)
    print(f"Loaded cached curves <- {cache_path}")
    return {
        "severities": data["severities"],
        "hf_noise_score": data["hf_noise_score"],
        "hf_noise_stderr": data["hf_noise_stderr"] if "hf_noise_stderr" in data else np.zeros_like(data["hf_noise_score"]),
        "wind_rotation_score": data["wind_rotation_score"],
        "wind_rotation_stderr": data["wind_rotation_stderr"] if "wind_rotation_stderr" in data else np.zeros_like(data["wind_rotation_score"]),
        "forecast_results": forecast_results,
        "score_kind": score_kind,
        "test_real_ranges": cached_test_real_ranges,
        "real_file": cached_real_file,
        "real_term": float(data["real_term"]) if "real_term" in data else np.nan,
        "real_term_stderr": float(data["real_term_stderr"]) if "real_term_stderr" in data else np.nan,
        "pangu_model_path": str(data["pangu_model_path"]) if "pangu_model_path" in data else "",
        "pangu_file": str(data["pangu_file"]) if "pangu_file" in data else "",
    }


def style_axis(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors=INK_BLACK, labelsize=18, width=1.0)
    for spine in ax.spines.values():
        spine.set_edgecolor(INK_BLACK)
        spine.set_linewidth(1.2)
    ax.grid(True, which="both", alpha=0.25, color=INK_BLACK, linewidth=0.6)


def add_zero_reference(ax):
    ax.axhline(0.0, color=INK_BLACK, linewidth=1.0, alpha=0.35)


def set_y_scale(ax, y_scale):
    if y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=1e-4, linscale=0.5)
    elif y_scale != "linear":
        raise ValueError(f"Unknown y_scale='{y_scale}'")


def resolve_npz_path(path):
    path = Path(path)
    if path.exists():
        return path
    fallback = Path(path.name)
    if fallback.exists():
        return fallback
    return path


def draw_missing_panel(ax, title, path):
    style_axis(ax)
    ax.set_title(title, fontsize=26, color=INK_BLACK, fontweight="bold", pad=14)
    ax.text(
        0.5, 0.5, f"Missing data:\n{path}",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=16, color=INK_BLACK,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def draw_fid_severity_panel(ax, sev_npz_path):
    sev_npz_path = resolve_npz_path(sev_npz_path)
    if not sev_npz_path.exists():
        draw_missing_panel(ax, "I-JEPA Sensitivity to Synthetic Corruptions", sev_npz_path)
        return

    d = np.load(sev_npz_path)
    severities = d["severities"]
    fid_noise = np.maximum(d["fid_noise"], 0.0)
    fid_rotation = np.maximum(d["fid_rotation"], 0.0)

    style_axis(ax)
    ax.plot(
        severities, fid_noise,
        marker="o", markersize=11, lw=3.0,
        color=YALE_BLUE, label="High-Freq Noise",
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )
    ax.plot(
        severities, fid_rotation,
        marker="s", markersize=11, lw=3.0,
        color=BLUSH_PINK, label="Wind-Vector Rotation",
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelcolor="#999999", labelsize=14)
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Corruption Severity", fontsize=22, color=INK_BLACK, labelpad=10)
    ax.set_ylabel("Generalized FID (symlog)", fontsize=22, color=INK_BLACK, labelpad=10)
    legend = ax.legend(
        fontsize=20, loc="lower right",
        frameon=True, facecolor="white", edgecolor=INK_BLACK,
        labelcolor=INK_BLACK,
    )
    legend.get_frame().set_linewidth(1.0)


def draw_fid_leadtime_panel(ax, lt_npz_path):
    lt_npz_path = resolve_npz_path(lt_npz_path)
    if not lt_npz_path.exists():
        draw_missing_panel(ax, "Held-Out Forecast Divergence from ERA5", lt_npz_path)
        return

    d = np.load(lt_npz_path)
    leads = d["lead_times_hours"]
    fid_ifs_hres = np.maximum(d["fid_ifs_hres"], 0.0)
    fid_graphcast = np.maximum(d["fid_graphcast"], 0.0)
    fid_pangu = np.maximum(d["fid_pangu"], 0.0) if "fid_pangu" in d.files else None

    style_axis(ax)
    ax.plot(
        leads, fid_ifs_hres,
        marker="^", markersize=11, lw=3.0,
        color=FORECAST_COLORS["IFS HRES"], label=FORECAST_DISPLAY_LABELS["IFS HRES"],
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )
    ax.plot(
        leads, fid_graphcast,
        marker="D", markersize=11, lw=3.0,
        color=FORECAST_COLORS["GraphCast"], label="GraphCast",
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )
    if fid_pangu is not None:
        ax.plot(
            leads, fid_pangu,
            marker="v", markersize=11, lw=3.0,
            color=FORECAST_COLORS["Pangu-Weather"], label="Pangu-Weather",
            markeredgecolor=INK_BLACK, markeredgewidth=0.8,
        )
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelcolor="#999999", labelsize=14)
    ax.set_xscale("log")
    ax.set_xticks(list(leads))
    ax.set_xticklabels([str(int(x)) for x in leads])
    ax.minorticks_off()
    ax.set_xlim(leads.min() * 0.85, leads.max() * 1.15)
    ax.set_xlabel("Forecast Lead Time (hours)", fontsize=22, color=INK_BLACK, labelpad=10)
    ax.set_ylabel("Generalized FID (symlog)", fontsize=22, color=INK_BLACK, labelpad=10)
    legend = ax.legend(
        fontsize=20, loc="lower right",
        frameon=True, facecolor="white", edgecolor=INK_BLACK,
        labelcolor=INK_BLACK,
    )
    legend.get_frame().set_linewidth(1.0)


def save_figure(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    if output_path.suffix.lower() == ".pdf":
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {output_path}\nSaved -> {png_path}")
    else:
        print(f"Saved -> {output_path}")
    plt.close(fig)


def plot_figure(curves, output_path, y_scale="linear", fid_severity_npz=None, fid_leadtime_npz=None):
    fig_sq, axes = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    ax_top, ax_fid_top = axes[0]
    ax_bot, ax_fid_bot = axes[1]
    fig_sq.patch.set_facecolor("white")
    fig_sq.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, hspace=0.12, wspace=0.08)

    style_axis(ax_top)
    ax_top.errorbar(
        curves["severities"], curves["hf_noise_score"], yerr=curves["hf_noise_stderr"],
        marker="o", markersize=11, lw=3.0, capsize=4,
        color=YALE_BLUE, label="High-Freq Noise",
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )
    ax_top.errorbar(
        curves["severities"], curves["wind_rotation_score"], yerr=curves["wind_rotation_stderr"],
        marker="s", markersize=11, lw=3.0, capsize=4,
        color=BLUSH_PINK, label="Wind-Vector Rotation",
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )
    set_y_scale(ax_top, "linear")
    ax_top.tick_params(axis="y", labelleft=True, labelcolor=INK_BLACK, labelsize=18)
    ax_top.set_xlim(0.0, 1.05)
    ax_top.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_top.set_xlabel("Corruption Severity", fontsize=22, color=INK_BLACK, labelpad=10)
    ax_top.set_ylabel(r"$S_{\mathrm{adv}}(Q)$", fontsize=22, color=INK_BLACK, labelpad=10)
    style_axis(ax_bot)
    for label, (leads, values, stderr) in curves["forecast_results"].items():
        ax_bot.errorbar(
            leads, values, yerr=stderr,
            marker=FORECAST_MARKERS.get(label, "o"), markersize=11, lw=3.0, capsize=4,
            color=FORECAST_COLORS.get(label, INK_BLACK),
            label=FORECAST_DISPLAY_LABELS.get(label, label),
            markeredgecolor=INK_BLACK, markeredgewidth=0.8,
        )
    set_y_scale(ax_bot, "linear")
    ax_bot.tick_params(axis="y", labelcolor=INK_BLACK, labelsize=18)
    ax_bot.set_xscale("log")
    if curves["forecast_results"]:
        all_leads = np.concatenate([v[0] for v in curves["forecast_results"].values()])
        lead_ticks = sorted(set(int(x) for x in all_leads))
        ax_bot.set_xticks(lead_ticks)
        ax_bot.set_xticklabels([str(x) for x in lead_ticks])
        ax_bot.set_xlim(all_leads.min() * 0.85, all_leads.max() * 1.15)
    ax_bot.minorticks_off()
    ax_bot.set_xlabel("Forecast Lead Time (hours)", fontsize=22, color=INK_BLACK, labelpad=10)
    ax_bot.set_ylabel(r"$S_{\mathrm{adv}}(Q)$", fontsize=22, color=INK_BLACK, labelpad=10)
    if fid_severity_npz is not None:
        draw_fid_severity_panel(ax_fid_top, fid_severity_npz)
    else:
        ax_fid_top.axis("off")
    if fid_leadtime_npz is not None:
        draw_fid_leadtime_panel(ax_fid_bot, fid_leadtime_npz)
    else:
        ax_fid_bot.axis("off")

    ax_top.set_title("Discriminator", fontsize=28, color=INK_BLACK, fontweight="bold", pad=18)
    ax_fid_top.set_title("I-JEPA", fontsize=28, color=INK_BLACK, fontweight="bold", pad=18)
    fig_sq.text(
        -0.015, 0.75, "Held-Out Corruption",
        ha="center", va="center", rotation="vertical",
        fontsize=28, color=INK_BLACK, fontweight="bold",
    )
    fig_sq.text(
        -0.015, 0.25, "Held-Out Forecast",
        ha="center", va="center", rotation="vertical",
        fontsize=28, color=INK_BLACK, fontweight="bold",
    )

    save_figure(fig_sq, output_path)


def load_cfg():
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="conf"):
        return compose(config_name="config")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-file", default=None)
    parser.add_argument("--ifs-file", default=None)
    parser.add_argument("--graphcast-file", default=None)
    parser.add_argument("--pangu-file", default=None)
    parser.add_argument("--pangu-holdout-model", default=None)
    parser.add_argument("--n-severity-steps", type=int, default=7)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/poster_discriminator_reverse_kl.pdf")
    parser.add_argument("--cache", default="results/poster_discriminator_reverse_kl_data.npz")
    parser.add_argument(
        "--fid-severity-npz",
        type=Path,
        default=Path("poster_fid_severity_data_nontemporal.npz"),
    )
    parser.add_argument(
        "--fid-leadtime-npz",
        type=Path,
        default=Path("poster_fid_leadtime_data_nontemporal.npz"),
    )
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--y-scale", choices=("symlog", "linear"), default="linear")
    args = parser.parse_args()

    cfg = load_cfg()
    validate_no_train_test_overlap(cfg)
    real_file = str(Path(args.real_file or cfg.get("test_real_nc_file", DEFAULT_REAL)))
    variables = variables_from_config(cfg)
    variable_tag = cfg.get("selected_variable", "all_fields") if len(variables) == 1 else "all_fields"
    pangu_model_path = Path(
        args.pangu_holdout_model
        or Path(cfg.output_dir) / f"weather_discriminator_{cfg.model_name}_{variable_tag}_pangu_holdout_lightning.pth"
    )
    cache_path = Path(args.cache)
    if cache_path.exists() and not args.recompute:
        curves = load_cache(cache_path, cfg.test_real_ranges, real_file)
        if curves is not None and pangu_model_path.exists() and "Pangu-Weather" not in curves["forecast_results"]:
            print(f"Ignoring cache without Pangu-Weather curve <- {cache_path}")
            curves = None
        if curves is None:
            curves = compute_curves(args, cfg)
            save_cache(cache_path, curves)
    else:
        curves = compute_curves(args, cfg)
        save_cache(cache_path, curves)

    plot_figure(
        curves,
        args.output,
        args.y_scale,
        fid_severity_npz=args.fid_severity_npz,
        fid_leadtime_npz=args.fid_leadtime_npz,
    )


if __name__ == "__main__":
    main()
