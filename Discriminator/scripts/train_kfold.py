import os
import re
import signal
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import hydra
from omegaconf import DictConfig


NUMERICAL_MODELS = {"IFS HRES", "ERA5 Forecast"}
SCRIPT_DIR = Path(__file__).resolve().parent


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
    """Return the checkpoint filename tag for configured input channels."""
    variables = variables_from_config(cfg)
    return cfg.selected_variable if len(variables) == 1 else "all_fields"


def model_key(name):
    """Normalize display names and filenames into coarse model keys."""
    normalized = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "graphcast": "GraphCast",
        "pangu": "Pangu-Weather",
        "panguweather": "Pangu-Weather",
        "fuxi": "FuXi",
        "keisler": "Keisler",
        "neuralgcm": "NeuralGCM",
        "sphericalcnn": "SphericalCNN",
        "ifshres": "IFS HRES",
        "era5forecast": "ERA5 Forecast",
        "era5forecast6steps": "ERA5 Forecast",
    }
    for token, key in aliases.items():
        if token in normalized:
            return key
    return name


def configured_training_files(cfg, holdout_model=None):
    """Return all configured neural forecast files except the holdout model."""
    train_files = []
    configured_files = cfg.get("fake_nc_file", [])
    if isinstance(configured_files, str):
        configured_files = [configured_files]

    for path in configured_files:
        key = model_key(os.path.basename(path))
        if key in NUMERICAL_MODELS:
            continue
        if holdout_model is not None and key == holdout_model:
            continue
        if os.path.exists(path):
            train_files.append(path)
        else:
            print(f"Warning: configured training file not found, skipping: {path}")

    return train_files


def require_train_files(train_files, context):
    """Fail before launching a child trainer with an empty fake file list."""
    if train_files:
        return
    raise RuntimeError(
        f"No fake training files resolved for {context}. Check cfg.comparison_files "
        "against the files that exist under DATA_DIR."
    )


def as_time_ranges(value):
    """Normalize `[start, end]` or `[[start, end], ...]` into ranges."""
    value = list(value or [])
    if len(value) == 2 and all(isinstance(item, str) for item in value):
        return [value]
    return [list(item) for item in value]


def hydra_ranges_arg(ranges):
    """Format one or more date ranges as a compact Hydra list override."""
    ranges = as_time_ranges(ranges)
    formatted = ["[" + ",".join(f"'{value}'" for value in time_range) + "]" for time_range in ranges]
    return "[" + ",".join(formatted) + "]"


def parse_date(value):
    """Parse the YYYY-MM-DD dates used in NetCDF filenames and config ranges."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def format_date(value):
    """Format a date object for Hydra range overrides."""
    return value.isoformat()


def file_timeframe(path):
    """Extract the inclusive date span from a forecast filename."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.nc$", os.path.basename(path))
    if not match:
        raise ValueError(f"Could not parse date range from forecast filename: {path}")
    return [match.group(1), match.group(2)]


def merge_time_ranges(ranges):
    """Merge overlapping or adjacent date ranges."""
    parsed = sorted((parse_date(start), parse_date(end)) for start, end in as_time_ranges(ranges))
    if not parsed:
        return []

    merged = [parsed[0]]
    for start, end in parsed[1:]:
        current_start, current_end = merged[-1]
        if start <= current_end + timedelta(days=1):
            merged[-1] = (current_start, max(current_end, end))
        else:
            merged.append((start, end))
    return [[format_date(start), format_date(end)] for start, end in merged]


def complement_time_ranges(base_ranges, excluded_ranges):
    """Return portions of `base_ranges` outside the excluded date ranges."""
    excluded = [(parse_date(start), parse_date(end)) for start, end in merge_time_ranges(excluded_ranges)]
    complement = []

    for base_start_text, base_end_text in as_time_ranges(base_ranges):
        cursor = parse_date(base_start_text)
        base_end = parse_date(base_end_text)

        for excluded_start, excluded_end in excluded:
            if excluded_end < cursor or excluded_start > base_end:
                continue
            if cursor < excluded_start:
                complement.append([format_date(cursor), format_date(excluded_start - timedelta(days=1))])
            cursor = max(cursor, excluded_end + timedelta(days=1))
            if cursor > base_end:
                break

        if cursor <= base_end:
            complement.append([format_date(cursor), format_date(base_end)])

    return complement


def fake_file_union_ranges(train_files):
    """Return the merged date union covered by the fake files in one subrun."""
    return merge_time_ranges(file_timeframe(path) for path in train_files)


def kfold_time_ranges(cfg, train_files):
    """Return fake model-time union and ERA5 complement for one k-fold subrun."""
    fake_union_ranges = fake_file_union_ranges(train_files)
    real_available_range = [file_timeframe(cfg.real_nc_file)]
    real_complement_ranges = complement_time_ranges(real_available_range, fake_union_ranges)
    return fake_union_ranges, real_complement_ranges


def kfold_checkpoint_dir(cfg):
    """Return the directory that stores k-fold discriminator weights."""
    return cfg.get("kfold_checkpoint_dir", os.path.join(cfg.output_dir, "kfold_checkpoints"))


def build_train_command(cfg, train_files, output_filename):
    """Build a Hydra override command for one discriminator training run."""
    train_files_arg = "[" + ",".join(train_files) + "]"
    fake_union_ranges, real_complement_ranges = kfold_time_ranges(cfg, train_files)
    return [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "train_discriminator.py"),
        "--config-name",
        cfg.get("child_config_name", "kfold_config"),
        f"++fake_nc_file={train_files_arg}",
        f"++selected_variable={cfg.selected_variable}",
        f"++model_name={cfg.model_name}",
        f"++output_filename={output_filename}",
        f"++output_dir={kfold_checkpoint_dir(cfg)}",
        f"++project_name={cfg.project_name}_kfold",
        f"++epochs={cfg.epochs}",
        f"++logger={cfg.get('logger', 'csv')}",
        f"++batch_size={cfg.batch_size}",
        f"++num_workers={cfg.num_workers}",
        f"++max_samples={cfg.get('max_samples', 0)}",
        f"++precision={cfg.precision}",
        f"++train_fake_range={hydra_ranges_arg(fake_union_ranges)}",
        f"++train_real_range={hydra_ranges_arg(real_complement_ranges)}",
        f"++test_fake_range={hydra_ranges_arg(fake_union_ranges)}",
        f"++test_real_ranges={hydra_ranges_arg(fake_union_ranges)}",
        "++skip_train_test_overlap_check=true",
        "++augment=true",
    ]


def describe_child_failure(exc):
    """Return a concise explanation for child trainer failures."""
    if exc.returncode is None or exc.returncode >= 0:
        return str(exc)

    signal_number = -exc.returncode
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = f"signal {signal_number}"

    message = f"child trainer was killed by {signal_name}"
    if signal_name == "SIGKILL":
        message += (
            ". On Slurm this usually means the job hit its time or memory limit; "
            "try a longer allocation for full training, or use max_samples/epochs/"
            "batch_size/num_workers overrides for a smoke run."
        )
    return message


@hydra.main(version_base=None, config_path="../conf", config_name="kfold_config")
def main(cfg: DictConfig):
    """Train leave-one-neural-model-out and full-pool discriminators."""
    comparison_files = cfg.comparison_files
    models = list(comparison_files.keys())
    checkpoint_dir = kfold_checkpoint_dir(cfg)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Avoid retraining identical pools created by different holdout requests.
    trained_models = {}  # key: tuple(train_files), value: output filename

    neural_holdouts = [model for model in models if model not in NUMERICAL_MODELS]
    numerical_models = [model for model in models if model in NUMERICAL_MODELS]

    for model in numerical_models:
        print(f"Skipping holdout discriminator for Numerical Model: {model}")

    # Phase 1: train one discriminator per neural model holdout.
    for model_to_exclude in neural_holdouts:
        print(f"\n" + "="*50)
        print(f"Target Model for Evaluation: {model_to_exclude}")
        print("="*50 + "\n")

        train_files = configured_training_files(cfg, holdout_model=model_to_exclude)

        if not train_files:
            print(f"Error: No training files available for evaluation of {model_to_exclude}. Skipping.")
            continue

        output_filename = (
            f"discriminator_{cfg.model_name}_{variable_tag(cfg)}_"
            f"exclude_{safe_model_name(model_to_exclude)}.pth"
        )

        train_files_key = tuple(sorted(train_files))

        if train_files_key in trained_models:
            existing_filename = trained_models[train_files_key]
            print(f"Training set identical to one used for {existing_filename}.")

            existing_path = os.path.join(checkpoint_dir, existing_filename)
            new_path = os.path.join(checkpoint_dir, output_filename)

            if os.path.exists(existing_path) and existing_filename != output_filename:
                print(f"Copying {existing_filename} to {output_filename}")
                shutil.copy2(existing_path, new_path)
            continue

        print(f"Training on AI Pool: {[os.path.basename(f) for f in train_files]}")
        fake_union_ranges, real_complement_ranges = kfold_time_ranges(cfg, train_files)
        print(f"Training fake ranges: {fake_union_ranges}")
        print(f"Training ERA5 ranges: {real_complement_ranges}")
        print(f"Testing ranges: {fake_union_ranges}")

        cmd = build_train_command(cfg, train_files, output_filename)

        print(f"Executing command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            print(f"Successfully trained and saved: {output_filename}")
            trained_models[train_files_key] = output_filename
        except subprocess.CalledProcessError as e:
            print(f"Failed to train for target model {model_to_exclude}: {describe_child_failure(e)}")

    # Phase 2: train one full neural pool for numerical-model evaluation.
    print(f"\n" + "="*50)
    print(f"Training Full AI Pool (for evaluating numerical models)")
    print("="*50 + "\n")

    train_files = configured_training_files(cfg)

    output_filename = f"discriminator_{cfg.model_name}_{variable_tag(cfg)}_full_pool.pth"
    train_files_key = tuple(sorted(train_files))
    require_train_files(train_files, "full-pool k-fold training")
    
    if train_files_key not in trained_models:
        fake_union_ranges, real_complement_ranges = kfold_time_ranges(cfg, train_files)
        print(f"Training fake ranges: {fake_union_ranges}")
        print(f"Training ERA5 ranges: {real_complement_ranges}")
        print(f"Testing ranges: {fake_union_ranges}")
        cmd = build_train_command(cfg, train_files, output_filename)
        print(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    else:
        existing_filename = trained_models[train_files_key]
        shutil.copy2(os.path.join(checkpoint_dir, existing_filename), os.path.join(checkpoint_dir, output_filename))
        print(f"Copied {existing_filename} to {output_filename} as Full Pool")

if __name__ == "__main__":
    main()
