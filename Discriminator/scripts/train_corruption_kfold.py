"""Train leave-one-corruption-group-out discriminators.

This experiment keeps the k-fold time split used by `train_kfold.py`, but the
fold axis is the synthetic corruption group instead of the forecast model. Each
checkpoint is trained with a configured list of corruptions removed from the
augmentation pool.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

try:
    from .train_kfold import (
        configured_training_files,
        describe_child_failure,
        hydra_ranges_arg,
        kfold_time_ranges,
        require_train_files,
        safe_model_name,
        variable_tag,
    )
except ImportError:
    from train_kfold import (
        configured_training_files,
        describe_child_failure,
        hydra_ranges_arg,
        kfold_time_ranges,
        require_train_files,
        safe_model_name,
        variable_tag,
    )


SCRIPT_DIR = Path(__file__).resolve().parent


def corruption_kfold_checkpoint_dir(cfg):
    """Return the directory for leave-one-corruption-out checkpoints."""
    return cfg.get(
        "corruption_kfold_checkpoint_dir",
        os.path.join(cfg.output_dir, "corruption_kfold_checkpoints"),
    )


def hydra_string_list_arg(values):
    """Format string values as a compact Hydra list override."""
    return "[" + ",".join(f"'{value}'" for value in values) + "]"


def corruption_kfold_types(cfg):
    """Return the full corruption pool for the experiment."""
    values = cfg.get("corruption_kfold_types", cfg.get("corruption_types", []))
    values = [str(value) for value in values]
    if len(values) < 2:
        raise ValueError("corruption_kfold_types must contain at least two corruption names.")
    return values


def as_corruption_group(value):
    """Normalize one configured holdout entry into a list of corruption names."""
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def corruption_kfold_holdouts(cfg):
    """Return configured held-out corruption groups.

    `corruption_kfold_holdouts` may contain either strings or lists. When it is
    omitted, the experiment falls back to true leave-one-corruption-out folds.
    """
    holdouts = cfg.get("corruption_kfold_holdouts")
    if not holdouts:
        return [[corruption] for corruption in corruption_kfold_types(cfg)]
    groups = [as_corruption_group(entry) for entry in holdouts]
    known = set(corruption_kfold_types(cfg))
    unknown = sorted({corruption for group in groups for corruption in group if corruption not in known})
    if unknown:
        raise ValueError(f"corruption_kfold_holdouts contains unknown corruptions: {unknown}")
    return groups


def corruption_group_tag(corruptions):
    """Build a stable filename tag for one held-out corruption group."""
    return "plus".join(safe_model_name(corruption) for corruption in corruptions)


def checkpoint_filename(cfg, heldout_corruptions):
    """Build the checkpoint filename for one held-out corruption group."""
    return (
        f"discriminator_{cfg.model_name}_{variable_tag(cfg)}_"
        f"corruption_exclude_{corruption_group_tag(heldout_corruptions)}.pth"
    )


def build_train_command(cfg, train_files, heldout_corruptions, output_filename):
    """Build a child train_discriminator.py command for one corruption fold."""
    heldout_set = set(heldout_corruptions)
    train_corruptions = [
        corruption for corruption in corruption_kfold_types(cfg)
        if corruption not in heldout_set
    ]
    if not train_corruptions:
        raise ValueError(f"Held-out group leaves no training corruptions: {heldout_corruptions}")
    train_fake_ranges, train_real_ranges, test_ranges = kfold_time_ranges(cfg, train_files)
    train_files_arg = "[" + ",".join(train_files) + "]"

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
        f"++output_dir={corruption_kfold_checkpoint_dir(cfg)}",
        f"++project_name={cfg.project_name}_corruption_kfold",
        f"++epochs={cfg.epochs}",
        f"++logger={cfg.get('logger', 'csv')}",
        f"++batch_size={cfg.batch_size}",
        f"++num_workers={cfg.num_workers}",
        f"++max_samples={cfg.get('max_samples', 0)}",
        f"++precision={cfg.precision}",
        f"++train_fake_range={hydra_ranges_arg(train_fake_ranges)}",
        f"++train_real_range={hydra_ranges_arg(train_real_ranges)}",
        f"++test_fake_range={hydra_ranges_arg(test_ranges)}",
        f"++test_real_ranges={hydra_ranges_arg(test_ranges)}",
        f"++corruption_types={hydra_string_list_arg(train_corruptions)}",
        "++allow_fake_train_test_overlap=true",
        "++augment=true",
    ]


@hydra.main(version_base=None, config_path="../conf", config_name="kfold_config")
def main(cfg: DictConfig):
    """Train one discriminator per held-out synthetic corruption group."""
    checkpoint_dir = corruption_kfold_checkpoint_dir(cfg)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_files = configured_training_files(cfg)
    require_train_files(train_files, "leave-one-corruption-out training")
    train_fake_ranges, train_real_ranges, test_ranges = kfold_time_ranges(cfg, train_files)

    print(f"Training on AI Pool: {[os.path.basename(path) for path in train_files]}")
    print(f"Training fake ranges: {train_fake_ranges}")
    print(f"Training ERA5 ranges: {train_real_ranges}")
    print(f"Testing ranges masked from ERA5 training: {test_ranges}")

    for heldout_corruptions in corruption_kfold_holdouts(cfg):
        output_filename = checkpoint_filename(cfg, heldout_corruptions)
        output_path = os.path.join(checkpoint_dir, output_filename)
        heldout_set = set(heldout_corruptions)
        train_corruptions = [
            corruption for corruption in corruption_kfold_types(cfg)
            if corruption not in heldout_set
        ]
        if not train_corruptions:
            raise ValueError(f"Held-out group leaves no training corruptions: {heldout_corruptions}")

        print("\n" + "=" * 50)
        print(f"Held-out corruptions: {heldout_corruptions}")
        print(f"Training corruptions: {train_corruptions}")
        print("=" * 50 + "\n")

        cmd = build_train_command(cfg, train_files, heldout_corruptions, output_filename)
        print(f"Executing command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            print(f"Successfully trained and saved: {output_filename}")
        except subprocess.CalledProcessError as exc:
            print(f"Failed to train for held-out corruptions {heldout_corruptions}: {describe_child_failure(exc)}")
            if os.path.exists(output_path):
                backup_path = output_path + ".failed"
                shutil.move(output_path, backup_path)
                print(f"Moved partial checkpoint to: {backup_path}")


if __name__ == "__main__":
    main()
