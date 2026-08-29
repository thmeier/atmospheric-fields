#!/bin/bash
#SBATCH --job-name=ucast_smoke
#SBATCH --account=pmlr_jobs
#SBATCH --partition=jobs
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=ucast_smoke_%j.out
#SBATCH --error=ucast_smoke_%j.err

# torch here is 2.5.1+cu121: supports sm_75 (2080ti), NOT sm_120 (5060ti).
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

TEAM=/cluster/courses/pmlr/teams/team07/data

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Settle the question: is team07 writable from THIS compute node?
if touch "$TEAM/.wtest_$SLURM_JOB_ID" 2>/dev/null; then
    echo "team07 from compute node: WRITABLE"
    rm -f "$TEAM/.wtest_$SLURM_JOB_ID"
    OUT="$TEAM/ucast_smoke_out.nc"
else
    echo "team07 from compute node: READ-ONLY (falling back to scratch)"
    OUT=/work/scratch/ddemler/ucast_cluster_smoke.nc
fi
echo "output -> $OUT"

cd ~/u-cast
python run_inference_standalone.py \
  --ckpt-path "$TEAM/ucast_weights/ucast.ckpt" \
  --config-path configs/config_inference.yaml \
  --data-dir "$TEAM/era5_ucast_smoke" \
  --ic-start-dates 2020-01-01T12 \
  --ensemble-size 2 \
  --prediction-horizon 4 \
  --score \
  --device cuda \
  --output-path "$OUT"
echo "SMOKE TEST OK"
