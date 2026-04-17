#!/bin/bash
#SBATCH --job-name=mae_era5_v1
#SBATCH --time=24:00:00          # Max runtime (e.g. 24 hours)
#SBATCH --account=pmlr_jobs      # Course project tag for long running batch jobs
#SBATCH --mem=64G                # Request 64GB of RAM
#SBATCH --gpus=1                 # Request exactly 1 GPU (e.g., --gpus=2080ti:1)
#SBATCH --output=slurm_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=slurm_%j.err     # Standard error log

# Mandatory module sourcing for this cluster
. /etc/profile.d/modules.sh

# Activate Conda Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

# Halt execution immediately if any python script crashes
set -eo pipefail

# Prevent Python from buffering print statements in SLURM
export PYTHONUNBUFFERED=1

# 1. Setup necessary directories on the cluster
mkdir -p checkpoints
mkdir -p /work/scratch/$USER/plots

echo "Starting ERA5 MAE Pipeline..."
echo "Running on node: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# 2. Run Pretraining (No --local flag means it defaults to the full cluster dataset)
echo -e "\n--- Phase 1: Training ---"
python train_mae.py --epochs 150 --batch-size 8 --num-workers 0 --stats-chunk-size 64 --lazy

# 3. Run Validation Protocol 1
echo -e "\n--- Phase 2: Probe Evaluation ---"
python eval_probe.py --batch-size 32 --num-workers 0 --lazy

# 4. Run Validation Protocol 2
echo -e "\n--- Phase 3: Distance Metrics Evaluation ---"
python eval_distances.py --batch-size 32 --num-workers 0 --lazy

echo "Pipeline finished successfully!"
