import os
import subprocess
import hydra
from omegaconf import DictConfig
import copy
import sys
import shutil

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    comparison_files = cfg.comparison_files
    models = list(comparison_files.keys())

    # Define numerical simulations that should ALWAYS be held out from training
    NUMERICAL_MODELS = ["IFS HRES", "ERA5 Forecast"]

    trained_models = {} # key: tuple of train_files, value: output_filename

    # --- Phase 1: Individual Neural Model Holdouts ---
    for model_to_exclude in models:
        # SKIP numerical models from having their own holdout discriminator
        if model_to_exclude in NUMERICAL_MODELS:
            print(f"Skipping holdout discriminator for Numerical Model: {model_to_exclude}")
            continue

        print(f"\n" + "="*50)
        print(f"Target Model for Evaluation: {model_to_exclude}")
        print("="*50 + "\n")

        train_files = []
        for m in models:
            # 1. Skip the target model (Standard k-fold)
            if m == model_to_exclude:
                continue

            # 2. ALWAYS skip numerical models from training pool
            if m in NUMERICAL_MODELS:
                continue

            path_2020 = comparison_files[m]
            # Replace 2020 range with 2018 range to get training files
            path_2018 = path_2020.replace("2020-01-01_2020-12-31", "2018-01-01_2018-12-31")

            if os.path.exists(path_2018):
                train_files.append(path_2018)
            elif os.path.exists(path_2020):
                print(f"Warning: 2018 data not found for {m}, using 2020 data instead.")
                train_files.append(path_2020)

        if not train_files:
            print(f"Error: No training files available for evaluation of {model_to_exclude}. Skipping.")
            continue

        # Unique output name
        safe_model_name = model_to_exclude.replace(' ', '_').replace('/', '_')
        output_filename = f"discriminator_{cfg.model_name}_{cfg.selected_variable}_exclude_{safe_model_name}.pth"

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

        # Build command for train_discriminator.py
        train_files_arg = "[" + ",".join(train_files) + "]"

        cmd = [
            sys.executable, "train_discriminator.py",
            f"++fake_nc_file={train_files_arg}",
            f"++selected_variable={cfg.selected_variable}",
            f"++model_name={cfg.model_name}",
            f"++output_filename={output_filename}",
            f"++project_name={cfg.project_name}_kfold",
            f"++epochs={cfg.epochs}",
            "++train_fake_range=['2018-01-01','2020-12-31']",
            "++augment=true" # Ensure augmentation is on, but only standard types are used in train_discriminator.py
        ]

        print(f"Executing command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully trained and saved: {output_filename}")
            trained_models[train_files_key] = output_filename
        except subprocess.CalledProcessError as e:
            print(f"Failed to train for target model {model_to_exclude}: {e}")

    # --- Phase 2: Full Pool (for evaluating numerical models) ---
    print(f"\n" + "="*50)
    print(f"Training Full AI Pool (for evaluating numerical models)")
    print("="*50 + "\n")

    train_files = []
    for m in models:
        if m in NUMERICAL_MODELS:
            continue
        path_2020 = comparison_files[m]
        path_2018 = path_2020.replace("2020-01-01_2020-12-31", "2018-01-01_2018-12-31")
        if os.path.exists(path_2018):
            train_files.append(path_2018)
        elif os.path.exists(path_2020):
            train_files.append(path_2020)

    output_filename = f"discriminator_{cfg.model_name}_{cfg.selected_variable}_full_pool.pth"
    train_files_key = tuple(sorted(train_files))
    
    if train_files_key not in trained_models:
        train_files_arg = "[" + ",".join(train_files) + "]"
        cmd = [
            sys.executable, "train_discriminator.py",
            f"++fake_nc_file={train_files_arg}",
            f"++selected_variable={cfg.selected_variable}",
            f"++model_name={cfg.model_name}",
            f"++output_filename={output_filename}",
            f"++project_name={cfg.project_name}_kfold",
            f"++epochs={cfg.epochs}",
            "++train_fake_range=['2018-01-01','2020-12-31']",
            "++augment=true"
        ]
        print(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        existing_filename = trained_models[train_files_key]
        shutil.copy2(os.path.join(cfg.output_dir, existing_filename), os.path.join(cfg.output_dir, output_filename))
        print(f"Copied {existing_filename} to {output_filename} as Full Pool")

if __name__ == "__main__":
    main()
