#!/bin/bash
#SBATCH --job-name=atm_fields_full
#SBATCH --time=48:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=36G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr
set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields
mkdir -p checkpoints
mkdir -p plots

echo "Starting full pipeline on node: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA built:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

# ------ Phase 1: Train twins (controlled comparison) -------------------------
echo -e "\n--- Phase 1a: Training MAE twin (100 epochs) ---"
python train/train_twins.py \
    --model mae \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 2 \
    --early-stopping-patience 15 \
    --lazy

echo -e "\n--- Phase 1b: Training I-JEPA twin (100 epochs) ---"
python train/train_twins.py \
    --model ijepa \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 2 \
    --early-stopping-patience 15 \
    --lazy

# ------ Phase 2: Probe evaluation (both models) -------------------------------
echo -e "\n--- Phase 2: Probe Evaluation (MAE + I-JEPA) ---"
python eval/eval_probe.py \
    --model both \
    --twin \
    --batch-size 64 \
    --num-workers 2 \
    --n-probe-samples 2000 \
    --lazy

# ------ Phase 5: Distance metrics (both models) -------------------------------
echo -e "\n--- Phase 5: Distance Metrics Evaluation (MAE + I-JEPA) ---"
python eval/eval_distances.py \
    --model both \
    --twin \
    --batch-size 64 \
    --num-workers 2 \
    --n-severity-steps 15 \
    --lazy

echo -e "\nPipeline finished successfully!"
