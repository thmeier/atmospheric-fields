"""Train one discriminator per forecast model for temporal holdout analysis."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

try:
    from .temporal_holdout_utils import (
        checkpoint_filename,
        discover_temporal_pairs,
        temporal_checkpoint_dir,
    )
except ImportError:
    from temporal_holdout_utils import (
        checkpoint_filename,
        discover_temporal_pairs,
        temporal_checkpoint_dir,
    )


SCRIPT_DIR = Path(__file__).resolve().parent


def hydra_range_arg(time_range):
    """Format a two-date range as a compact Hydra list override."""
    return "[" + ",".join(f"'{value}'" for value in time_range) + "]"


def describe_child_failure(exc):
    """Return a concise explanation for child trainer failures."""
    if exc.returncode is None or exc.returncode >= 0:
        return str(exc)

    signal_number = -exc.returncode
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = f"signal {signal_number}"
    return f"child trainer was killed by {signal_name}"


def build_train_command(cfg, model_label, train_file, test_file):
    """Build a child train_discriminator.py command for one forecast model."""
    return [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "train_discriminator.py"),
        "--config-name",
        "config",
        f"++fake_nc_file=[{train_file}]",
        f"++test_fake_nc_file={test_file}",
        f"++selected_variable={cfg.selected_variable}",
        f"++model_name={cfg.model_name}",
        f"++output_filename={checkpoint_filename(cfg, model_label)}",
        f"++output_dir={temporal_checkpoint_dir(cfg)}",
        f"++project_name={cfg.project_name}_temporal_holdout",
        f"++epochs={cfg.epochs}",
        f"++logger={cfg.get('logger', 'csv')}",
        f"++batch_size={cfg.batch_size}",
        f"++num_workers={cfg.num_workers}",
        f"++max_samples={cfg.get('max_samples', 0)}",
        f"++precision={cfg.precision}",
        f"++train_fake_range={hydra_range_arg(cfg.train_fake_range)}",
        f"++test_fake_range={hydra_range_arg(cfg.test_fake_range)}",
        f"++augment={str(cfg.get('augment', True)).lower()}",
    ]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Train temporal holdout discriminators."""
    pairs = discover_temporal_pairs(cfg)
    checkpoint_dir = temporal_checkpoint_dir(cfg)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not pairs:
        raise RuntimeError("No temporal train/test forecast pairs found.")

    print(f"Temporal fake train range: {cfg.train_fake_range}")
    print(f"Temporal fake test range: {cfg.test_fake_range}")

    for model_label, files in pairs.items():
        output_path = checkpoint_dir / checkpoint_filename(cfg, model_label)
        print("\n" + "=" * 50)
        print(f"Temporal holdout model: {model_label}")
        print("=" * 50)
        print(f"Training fake file: {files['train']}")
        print(f"Test fake file: {files['test']}")
        print(f"Output checkpoint: {output_path}")

        cmd = build_train_command(cfg, model_label, files["train"], files["test"])
        print(f"Executing command: {' '.join(map(str, cmd))}")
        try:
            subprocess.run(cmd, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        except subprocess.CalledProcessError as exc:
            print(f"Failed to train {model_label}: {describe_child_failure(exc)}")


if __name__ == "__main__":
    main()
