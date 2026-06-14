import os
import shutil
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


NUMERICAL_MODELS = {"IFS HRES", "ERA5 Forecast"}
SCRIPT_DIR = Path(__file__).resolve().parent


def safe_model_name(name):
    """Convert display model names into filename-safe tags."""
    return name.replace(" ", "_").replace("/", "_")


def training_file_for_comparison(path_2020):
    """Prefer the matching 2018 file; fall back to the configured path."""
    path_2018 = path_2020.replace("2020-01-01_2020-12-31", "2018-01-01_2018-12-31")
    if os.path.exists(path_2018):
        return path_2018
    if os.path.exists(path_2020):
        return path_2020
    return None


def hydra_range_arg(time_range):
    """Format a two-date range as a compact Hydra list override."""
    return "[" + ",".join(f"'{value}'" for value in time_range) + "]"


def build_train_command(cfg, train_files, output_filename):
    """Build a Hydra override command for one discriminator training run."""
    train_files_arg = "[" + ",".join(train_files) + "]"
    return [
        sys.executable,
        str(SCRIPT_DIR / "train_discriminator.py"),
        "--config-name",
        "kfold_config",
        f"++fake_nc_file={train_files_arg}",
        f"++selected_variable={cfg.selected_variable}",
        f"++model_name={cfg.model_name}",
        f"++output_filename={output_filename}",
        f"++project_name={cfg.project_name}_kfold",
        f"++epochs={cfg.epochs}",
        f"++logger={cfg.get('logger', 'csv')}",
        f"++train_fake_range={hydra_range_arg(cfg.train_fake_range)}",
        "++augment=true",
    ]

@hydra.main(version_base=None, config_path="../conf", config_name="kfold_config")
def main(cfg: DictConfig):
    """Train leave-one-neural-model-out and full-pool discriminators."""
    comparison_files = cfg.comparison_files
    models = list(comparison_files.keys())

    # Avoid retraining identical pools created by different holdout requests.
    trained_models = {}  # key: tuple(train_files), value: output filename

    # Phase 1: train one discriminator per neural model holdout.
    for model_to_exclude in models:
        if model_to_exclude in NUMERICAL_MODELS:
            print(f"Skipping holdout discriminator for Numerical Model: {model_to_exclude}")
            continue

        print(f"\n" + "="*50)
        print(f"Target Model for Evaluation: {model_to_exclude}")
        print("="*50 + "\n")

        train_files = []
        for m in models:
            if m == model_to_exclude:
                continue
            if m in NUMERICAL_MODELS:
                continue

            training_file = training_file_for_comparison(comparison_files[m])
            if training_file is None:
                continue
            if training_file == comparison_files[m]:
                print(f"Warning: 2018 data not found for {m}, using 2020 data instead.")
            train_files.append(training_file)

        if not train_files:
            print(f"Error: No training files available for evaluation of {model_to_exclude}. Skipping.")
            continue

        output_filename = (
            f"discriminator_{cfg.model_name}_{cfg.selected_variable}_"
            f"exclude_{safe_model_name(model_to_exclude)}.pth"
        )

        train_files_key = tuple(sorted(train_files))

        if train_files_key in trained_models:
            existing_filename = trained_models[train_files_key]
            print(f"Training set identical to one used for {existing_filename}.")

            existing_path = os.path.join(cfg.output_dir, existing_filename)
            new_path = os.path.join(cfg.output_dir, output_filename)

            if os.path.exists(existing_path) and existing_filename != output_filename:
                print(f"Copying {existing_filename} to {output_filename}")
                shutil.copy2(existing_path, new_path)
            continue

        print(f"Training on AI Pool: {[os.path.basename(f) for f in train_files]}")

        cmd = build_train_command(cfg, train_files, output_filename)

        print(f"Executing command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully trained and saved: {output_filename}")
            trained_models[train_files_key] = output_filename
        except subprocess.CalledProcessError as e:
            print(f"Failed to train for target model {model_to_exclude}: {e}")

    # Phase 2: train one full neural pool for numerical-model evaluation.
    print(f"\n" + "="*50)
    print(f"Training Full AI Pool (for evaluating numerical models)")
    print("="*50 + "\n")

    train_files = []
    for m in models:
        if m in NUMERICAL_MODELS:
            continue
        training_file = training_file_for_comparison(comparison_files[m])
        if training_file is not None:
            train_files.append(training_file)

    output_filename = f"discriminator_{cfg.model_name}_{cfg.selected_variable}_full_pool.pth"
    train_files_key = tuple(sorted(train_files))
    
    if train_files_key not in trained_models:
        cmd = build_train_command(cfg, train_files, output_filename)
        print(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        existing_filename = trained_models[train_files_key]
        shutil.copy2(os.path.join(cfg.output_dir, existing_filename), os.path.join(cfg.output_dir, output_filename))
        print(f"Copied {existing_filename} to {output_filename} as Full Pool")

if __name__ == "__main__":
    main()
