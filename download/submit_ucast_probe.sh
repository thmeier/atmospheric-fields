#!/bin/bash
#SBATCH --job-name=ucast_probe
#SBATCH --account=pmlr_jobs
#SBATCH --partition=jobs
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=ucast_probe_%j.out
#SBATCH --error=ucast_probe_%j.err

# GPU-memory gate before the multi-hour 2020 run.
#
# Only 2 ensemble members have ever run on this 11 GB card. The production run wants 5.
# Arithmetic says it should fit (3.58 GB fp32 weights + fp16 activations under upstream's
# autocast), but "should" is not good enough to spend three GPU-hours on: this probe
# measures torch.cuda.max_memory_allocated() in about two minutes instead.
#
# Note the 2080ti pin: the cluster pmlr env has torch 2.5.1+cu121, whose arch list stops
# at sm_90, so the sm_120 5060ti nodes cannot run it at all.

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

TEAM=/cluster/courses/pmlr/teams/team07/data
REPO=~/atmospheric-fields

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for N in 2 5; do
    echo "───────── ensemble_size=${N} ─────────"
    python "$REPO/ucast/run_ucast_2020.py" \
      --ucast-repo ~/u-cast \
      --ckpt-path "$TEAM/ucast_weights/ucast.ckpt" \
      --data-glob "$TEAM/era5_ucast_2020/*.zarr" \
      --output-dir "/work/scratch/ddemler/ucast_probe_e${N}" \
      --prefix "probe_e${N}" \
      --start 2020-01-01 --end 2020-01-01T00 \
      --prediction-horizon 2 --ensemble-size "$N" \
      --lead-hours 12 24 \
      --chunk-size 1 --device cuda
done

echo "PROBE OK — compare the reported 'Peak CUDA memory' against the 11 GB card"
