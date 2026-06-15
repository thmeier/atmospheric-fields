#!/bin/bash
#SBATCH --job-name=probe_eval
#SBATCH --time=02:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=16G
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

mkdir -p plots

echo "Starting probe eval on node: $(hostname)"
nvidia-smi

python eval/eval_probe.py \
    --model both \
    --ijepa-size small \
    --batch-size 64 \
    --num-workers 2 \
    --n-probe-samples 2000 \
    --lazy

echo "Probe eval finished!"
