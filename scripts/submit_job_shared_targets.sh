#!/bin/bash
#SBATCH --job-name=atm_ablation_suite
#SBATCH --time=48:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=36G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=logs/ablation_suite_%j.out
#SBATCH --error=logs/ablation_suite_%j.err

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr
set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields
mkdir -p checkpoints
mkdir -p plots
mkdir -p logs

echo "Starting ablation suite on node: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA built:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

echo -e "\n=== Experiment A: Shared target blocks (MAE vs I-JEPA) ==="

echo -e "\n--- Phase A1: Training MAE twin with shared target blocks ---"
python train/train_twins.py \
    --model mae \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 2 \
    --mae-mask-mode target-blocks \
    --early-stopping-patience 15 \
    --lazy

echo -e "\n--- Phase A2: Training I-JEPA twin with shared target blocks ---"
python train/train_twins.py \
    --model ijepa \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 2 \
    --target-mask-mode shared-blocks \
    --early-stopping-patience 15 \
    --lazy

echo -e "\n--- Phase A3: Probe Evaluation ---"
python eval/eval_probe.py \
    --model both \
    --mae-variant shared-targets \
    --ijepa-variant shared-targets \
    --batch-size 64 \
    --num-workers 2 \
    --n-probe-samples 2000 \
    --lazy

echo -e "\n--- Phase A4: Distance Metrics Evaluation ---"
python eval/eval_distances.py \
    --model both \
    --mae-variant shared-targets \
    --ijepa-variant shared-targets \
    --batch-size 64 \
    --num-workers 2 \
    --n-severity-steps 15 \
    --lazy

echo -e "\n=== Experiment B: I-JEPA baseline vs world-band context ==="

echo -e "\n--- Phase B1: Training baseline I-JEPA twin ---"
python train/train_twins.py \
    --model ijepa \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 2 \
    --early-stopping-patience 15 \
    --lazy

echo -e "\n--- Phase B2: Training world-band I-JEPA twin ---"
python train/train_twins.py \
    --model ijepa \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 2 \
    --ijepa-context-mode world-band \
    --early-stopping-patience 15 \
    --lazy

echo -e "\n--- Phase B3a: Probe Evaluation (baseline) ---"
python eval/eval_probe.py \
    --model ijepa \
    --batch-size 64 \
    --num-workers 2 \
    --n-probe-samples 2000 \
    --lazy

echo -e "\n--- Phase B3b: Probe Evaluation (world-band) ---"
python eval/eval_probe.py \
    --model ijepa \
    --ijepa-variant world-band \
    --batch-size 64 \
    --num-workers 2 \
    --n-probe-samples 2000 \
    --lazy

echo -e "\n--- Phase B4a: Distance Metrics Evaluation (baseline) ---"
python eval/eval_distances.py \
    --model ijepa \
    --batch-size 64 \
    --num-workers 2 \
    --n-severity-steps 15 \
    --lazy

echo -e "\n--- Phase B4b: Distance Metrics Evaluation (world-band) ---"
python eval/eval_distances.py \
    --model ijepa \
    --ijepa-variant world-band \
    --batch-size 64 \
    --num-workers 2 \
    --n-severity-steps 15 \
    --lazy

echo -e "\nAblation suite finished successfully."
