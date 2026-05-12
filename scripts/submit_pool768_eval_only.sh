#!/bin/bash
#SBATCH --job-name=pool768_eval
#SBATCH --time=00:30:00
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

WORKERS=2
N_SAMPLES=500

echo "Node: $(hostname)"

echo
echo "============================================================"
echo "Eval: embed_dim=384  +  concat pooling (768 effective)"
echo "============================================================"
python eval/eval_real_vs_forecast.py \
    --mae-only \
    --pooling concat \
    --embed-dim 384 \
    --n-samples $N_SAMPLES \
    --batch-size 64 \
    --num-workers $WORKERS \
    --output-dir results/pool768_concat384

echo
echo "============================================================"
echo "Eval: embed_dim=768  +  max pooling"
echo "============================================================"
python eval/eval_real_vs_forecast.py \
    --mae-only \
    --pooling max \
    --embed-dim 768 \
    --n-samples $N_SAMPLES \
    --batch-size 64 \
    --num-workers $WORKERS \
    --output-dir results/pool768_max768

echo
echo "============================================================"
echo "Done. Results in:"
echo "  results/pool768_concat384/plots/real_vs_forecast/"
echo "  results/pool768_max768/plots/real_vs_forecast/"
echo "============================================================"
