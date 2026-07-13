"""Utilities for per-model temporal holdout discriminator experiments."""

import os
import re
from pathlib import Path


DATE_SPAN_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.nc$")
MODEL_ALIASES = {
    "graphcast": "GraphCast",
    "pangu": "Pangu-Weather",
    "panguweather": "Pangu-Weather",
    "fuxi": "FuXi",
    "keisler": "Keisler",
    "neuralgcm": "NeuralGCM",
    "sphericalcnn": "SphericalCNN",
    "ifshres": "IFS HRES",
    "era5forecast": "ERA5 Forecast",
}


def safe_model_name(name):
    """Convert display model names into filename-safe tags."""
    return name.replace(" ", "_").replace("/", "_")


def variables_from_config(cfg):
    """Return all configured input fields, falling back to one selected field."""
    variables = cfg.get("variables")
    if variables:
        return list(variables)
    return [cfg.selected_variable]


def variable_tag(cfg):
    """Return the checkpoint/output filename tag for configured input channels."""
    variables = variables_from_config(cfg)
    return cfg.selected_variable if len(variables) == 1 else "all_fields"


def normalized_model_token(name):
    """Normalize display names and filename stems for coarse model matching."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def model_label_from_path(path):
    """Infer a display model label from a forecast filename."""
    stem = Path(path).name
    model_part = stem.split("_6steps", 1)[0]
    token = normalized_model_token(model_part)
    for alias, label in MODEL_ALIASES.items():
        if alias in token:
            return label
    return model_part.replace("_", " ").replace("-", " ").title()


def date_span_from_path(path):
    """Return `(start, end)` date strings encoded in a forecast filename."""
    match = DATE_SPAN_RE.search(Path(path).name)
    if match is None:
        return None
    return match.group(1), match.group(2)


def range_year(time_range):
    """Return the start year from a `[start, end]` date range."""
    return str(time_range[0])[:4]


def candidate_forecast_files(cfg):
    """Collect forecast files from config plus the data directory."""
    candidates = set()
    for value in cfg.get("fake_nc_file", []):
        candidates.add(str(value))
    for value in cfg.get("comparison_files", {}).values():
        candidates.add(str(value))

    data_dir = Path(cfg.data_dir)
    if data_dir.exists():
        for path in data_dir.glob("*.nc"):
            candidates.add(str(path))

    return sorted(candidates)


def temporal_checkpoint_dir(cfg):
    """Return the directory that stores temporal-holdout discriminator weights."""
    return Path(cfg.get("temporal_checkpoint_dir", Path(cfg.output_dir) / "temporal_holdout_checkpoints"))


def discover_temporal_pairs(cfg):
    """Find model-specific train/test forecast files for temporal holdout."""
    train_range, test_range = cfg.train_fake_range, cfg.test_fake_range
    train_year = range_year(train_range)
    test_year = range_year(test_range)
    by_model = {}

    for path in candidate_forecast_files(cfg):
        if not os.path.exists(path):
            continue
        span = date_span_from_path(path)
        if span is None:
            continue
        label = model_label_from_path(path)
        start_year = span[0][:4]
        if start_year not in (train_year, test_year):
            continue
        by_model.setdefault(label, {})[start_year] = path

    pairs = {}
    for label, files_by_year in sorted(by_model.items()):
        train_file = files_by_year.get(train_year)
        test_file = files_by_year.get(test_year)
        if train_file is not None and test_file is not None:
            pairs[label] = {"train": train_file, "test": test_file}
    return pairs


def checkpoint_filename(cfg, model_label):
    """Return the temporal-holdout checkpoint filename for one model."""
    return (
        f"discriminator_{cfg.model_name}_{variable_tag(cfg)}_"
        f"temporal_{safe_model_name(model_label)}.pth"
    )


def checkpoint_path(cfg, model_label):
    """Return the temporal-holdout checkpoint path for one model."""
    return temporal_checkpoint_dir(cfg) / checkpoint_filename(cfg, model_label)
