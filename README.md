# Metric for Realism of Atmospheric Fields

Project work for the course ETH AI Center Projects in Machine Learning Research 2026 at ETH Zurich.

## Table of Contents

- [Cluster Setup](#cluster-setup)
- [Local Setup](#local-setup)
- [Data](#data)
- [Usage](#usage)
- [Results](#results)

---

## Cluster Setup

### Access

Connect to the ETH VPN if off-campus, then SSH in:

```bash
ssh <your-eth-username>@student-cluster.inf.ethz.ch
```

### Install Conda

Each team member needs their own conda installation in their home directory:

```bash
bash /cluster/data/miniconda/Miniconda3-py312_25.11.1-1-Linux-x86_64.sh
```

Accept the license and confirm the install location (`~/miniconda3`). When asked to initialize conda, type `yes`. Then accept the Terms of Service:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Reload your shell:

```bash
source ~/.bashrc
```

You should now see `(base)` in your prompt.

### Create the Environment

```bash
conda create -n pmlr python=3.12 -y
conda activate pmlr
```

If the package install fails, add the conda-forge channel first:

```bash
conda config --add channels conda-forge
```

Then install dependencies:

```bash
conda install --file requirements.txt
```

### Team Shared Storage

The shared team directory is at `/cluster/courses/pmlr/teams/team07`. Data downloads go here — only download once, don't overwrite files others are using.

If a teammate gets a permission denied error on a file you created, grant them access:

```bash
# for a file
setfacl -m u:<their-eth-username>:rwx <file-path>

# for a directory (recursive)
setfacl -R -m u:<their-eth-username>:rwx <dir-path>
setfacl -d -m u:<their-eth-username>:rwx <dir-path>
```

### Clone the Repo

```bash
cd ~
git clone <repo-url> atmospheric-fields
```

---

## Local Setup

Create a new conda environment using Python 3.12:

```bash
conda create -n pmlr python=3.12 -y && conda activate pmlr
```

Install dependencies:

```bash
conda install --file requirements.txt
```

---

## Data

To download ERA5 data to the shared cluster directory, edit and submit the Slurm job:

```bash
sbatch ~/atmospheric-fields/start_download_er5.sh
```

The key settings in `start_download_er5.sh`:

```
         SOURCE : Google Cloud bucket URL (determines resolution)
     TIME_START : start date in YYYY-MM-DD format
       TIME_END : end date in YYYY-MM-DD format
      VARIABLES : which variables to select
OUTPUT_FILENAME : output filename (.nc)
     OUTPUT_DIR : save location — shared at /cluster/courses/pmlr/teams/team07/data
```

Current dataset: 1.5-degree ERA5, 2004–2023, 4 variables:
- `2m_temperature`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `mean_sea_level_pressure`

Monitor a running job:

```bash
squeue --me
tail -f ~/atmospheric-fields/logs/test_<job_id>.out
```

---

## Usage

### Training

Train the MAE model on the cluster:

```bash
sbatch submit_job.sh
```

Or run locally:

```bash
python train_mae.py --local
python train_mae.py --large-local  # 5-year local dataset
```

Checkpoints and data statistics (`data_mean.npy`, `data_std.npy`, `best_mae_model.pth`) are saved to `checkpoints/`.

### Evaluation

Both eval scripts support `--local`, `--large-local`, or no flag (cluster dataset). Use `--eager` to load the full dataset into memory upfront.

**Validation Protocol 1 — Linear Probe (severity regression):**

Trains a small MLP on frozen MAE latents to predict corruption severity. Reports R² per corruption type.

```bash
python eval_probe.py --large-local --eager
```

**Validation Protocol 2 — Fréchet & MMD Distances:**

Computes Fréchet Distance and MMD between the reference latent distribution and corrupted distributions across a severity ladder.

```bash
python eval_distances.py --large-local --eager
```

Plots are saved to `plots/`.

### Corruptions

`corruptions.py` defines four corruption types applied to input fields:

| Corruption | Description | Severity → parameter |
|---|---|---|
| Gaussian Blur | Spatial smoothing | severity → sigma ∈ [0, 1.125] |
| High-Freq Noise | Per-pixel iid Gaussian noise | severity → std ∈ [0, 0.25] |
| GRF Noise | Spatially correlated Gaussian Random Field noise | severity → std ∈ [0, 0.375] |
| Random Pixel Replace | Replaces random pixels with Gaussian samples | severity → replace prob ∈ [0, 0.3] |

### Visualisation

```bash
python visualize_corruptions.py --local
```

---

## Results

TODO
