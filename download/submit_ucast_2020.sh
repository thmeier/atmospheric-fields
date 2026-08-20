#!/bin/bash
#SBATCH --job-name=ucast2020
#SBATCH --account=pmlr_jobs
#SBATCH --partition=jobs
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=ucast2020_%j.out
#SBATCH --error=ucast2020_%j.err

# Full U-Cast inference over 2020: 732 ICs (00z/12z), horizon 20 (240 h), 5 members,
# keeping the 9 analysis fields at 7 lead times.
#
# Run download/submit_ucast_probe.sh FIRST — 5 members has not been validated on an
# 11 GB card, and only one job may be queued at a time on this cluster.
#
# Details that are load-bearing, each learned the hard way:
#   * 2080ti pin — cluster torch 2.5.1+cu121 stops at sm_90, so 5060ti (sm_120) is out.
#   * Checkpoint read from team07, never `hf:` — the HF Xet downloader buffers all
#     7.16 GB in RAM and gets the job killed (exit 120).
#   * Output goes to scratch, not team07: /cluster/courses is read-only from some
#     compute nodes (node13 read-only, node01 writable). Copy to team07 from a LOGIN
#     node afterwards, where it is always writable.
#   * --skip-existing makes this resumable: a re-submit picks up at the first missing
#     chunk rather than redoing hours of work.

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

TEAM=/cluster/courses/pmlr/teams/team07/data
REPO=~/atmospheric-fields
OUT=/work/scratch/ddemler/ucast2020_chunks

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$OUT"

python "$REPO/ucast/run_ucast_2020.py" \
  --ucast-repo ~/u-cast \
  --ckpt-path "$TEAM/ucast_weights/ucast.ckpt" \
  --data-glob "$TEAM/era5_ucast_2020/*.zarr" \
  --output-dir "$OUT" \
  --prefix ucast2020 \
  --start 2020-01-01 --end 2020-12-31T23:59:59 \
  --ic-hours 0 12 \
  --prediction-horizon 20 \
  --ensemble-size 5 \
  --lead-hours 12 24 48 72 96 192 240 \
  --chunk-size 50 \
  --skip-existing \
  --device cuda \
  --score \
  --wandb-project ucast-inference \
  --wandb-entity weather-realism-pmlr \
  --wandb-run-name "ucast2020-h20-e5-$(date +%Y%m%d)"

echo "───── converting to baseline schema ─────"
python "$REPO/ucast/convert_to_baseline_schema.py" \
  --chunks "$OUT/ucast2020_chunk*.nc" \
  --output-dir /work/scratch/ddemler/ucast2020_final \
  --tag ucast \
  --span 2020-01-01_2020-12-31

echo "RUN COMPLETE"
echo "Now, from a LOGIN node (compute nodes cannot write to team07):"
echo "  cp -r /work/scratch/ddemler/ucast2020_final/* $TEAM/"
