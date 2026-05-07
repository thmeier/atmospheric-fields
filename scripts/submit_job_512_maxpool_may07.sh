#!/bin/bash
#SBATCH --job-name=atm_512_maxpool
#SBATCH --time=08:00:00
#SBATCH --account=pmlr_jobs
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --output=logs/512_maxpool_%j.out
#SBATCH --error=logs/512_maxpool_%j.err

. /etc/profile.d/modules.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

set -eo pipefail
export PYTHONUNBUFFERED=1

cd ~/atmospheric-fields
mkdir -p logs

# ── Configuration ────────────────────────────────────────────────────────────
# Eval-only ablation: rerun Phases 3–5 of submit_job_512_may07.sh with max
# pooling instead of mean pooling in extract_features. Reuses the checkpoints
# trained by that job (training is unaffected by pooling choice — pooling only
# enters extract_features at eval time).
SRC_DIR="/work/scratch/${USER}/results/may_07_512_encoder"
OUTPUT_DIR="/work/scratch/${USER}/results/may_07_512_maxpool_eval"
EMBED_DIM=512
MODEL_SIZE="twin"

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: source dir not found: $SRC_DIR"
    echo "Wait for submit_job_512_may07.sh to finish before running this."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Symlink checkpoints + normalization stats from the may07 run so the eval
# scripts (which look in --output-dir for these) find them. We use symlinks
# instead of copies to avoid duplicating ~hundreds of MB of weights.
for f in \
    "best_mae_model_${MODEL_SIZE}_d${EMBED_DIM}.pth" \
    "best_ijepa_model_${MODEL_SIZE}_d${EMBED_DIM}.pth" \
    "data_mean.npy" \
    "data_std.npy"; do
    src="$SRC_DIR/$f"
    dst="$OUTPUT_DIR/$f"
    if [ ! -e "$src" ]; then
        echo "ERROR: missing source artifact: $src"
        exit 1
    fi
    ln -sfn "$src" "$dst"
done

# ── Switch extract_features to max pooling ───────────────────────────────────
export EXTRACT_FEATURES_POOLING=max

echo "Starting 512-encoder MAX-POOL eval on node: $(hostname)"
echo "Source (checkpoints): $SRC_DIR"
echo "Output (plots/results): $OUTPUT_DIR"
echo "Embed dim: $EMBED_DIM, Model size: $MODEL_SIZE"
echo "EXTRACT_FEATURES_POOLING=$EXTRACT_FEATURES_POOLING"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# ── Phase 3: Probe evaluation (corruption regression, both models) ───────────
echo -e "\n============================================================"
echo "Phase 3: Probe evaluation — corruption severity regression (max pool)"
echo "============================================================"
python eval/eval_probe.py \
    --model both \
    --mae-size "$MODEL_SIZE" \
    --ijepa-size "$MODEL_SIZE" \
    --mae-embed-dim "$EMBED_DIM" \
    --ijepa-embed-dim "$EMBED_DIM" \
    --batch-size 64 \
    --num-workers 2 \
    --n-probe-samples 2000 \
    --output-dir "$OUTPUT_DIR" \
    --lazy

# ── Phase 4: Distance metrics (FID/MMD vs corruptions, both models) ──────────
echo -e "\n============================================================"
echo "Phase 4: Distance metrics — FID/MMD vs corruption severity (max pool)"
echo "============================================================"
python eval/eval_distances.py \
    --model both \
    --mae-size "$MODEL_SIZE" \
    --ijepa-size "$MODEL_SIZE" \
    --mae-embed-dim "$EMBED_DIM" \
    --ijepa-embed-dim "$EMBED_DIM" \
    --batch-size 64 \
    --num-workers 2 \
    --n-severity-steps 15 \
    --output-dir "$OUTPUT_DIR" \
    --lazy

# ── Phase 5: Real vs Forecast (ERA5 vs Pangu / GraphCast, both models) ───────
echo -e "\n============================================================"
echo "Phase 5: Real vs Forecast — ERA5 vs Pangu / GraphCast (max pool)"
echo "============================================================"
python eval/eval_real_vs_forecast.py \
    --model-size "$MODEL_SIZE" \
    --embed-dim "$EMBED_DIM" \
    --n-samples 500 \
    --batch-size 32 \
    --num-workers 2 \
    --output-dir "$OUTPUT_DIR"

echo -e "\n============================================================"
echo "Max-pool eval finished successfully."
echo "Compare against mean-pool baseline at: $SRC_DIR/plots/"
echo "Max-pool plots/results: $OUTPUT_DIR/plots/"
echo "Pull with:"
echo "  scp -r ddemler@student-cluster.inf.ethz.ch:$OUTPUT_DIR ."
echo "============================================================"
