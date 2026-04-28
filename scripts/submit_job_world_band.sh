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
mkdir -p logs

echo "Delegating to scripts/submit_job_shared_targets.sh so both ablation tests run in one sequential job."
bash scripts/submit_job_shared_targets.sh
