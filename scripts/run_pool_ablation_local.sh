#!/bin/bash
# Local mean-vs-max pooling ablation against the may07 512-encoder checkpoints.
#
# Runs Phases 3-5 (probe, distances, real-vs-forecast) twice — once with mean
# pooling, once with max pooling — against the same local dataset and same
# checkpoints. All outputs land in results/may_07_512_encoder/plots/ with a
# _pool-{mode} suffix so the runs are directly comparable.
#
# The pre-existing cluster plots in that folder (no _pool-* suffix) are left
# untouched but are NOT directly comparable — they used the full 2004-2023
# dataset, while this script uses --local (1y) or --large-local (5y).

set -eo pipefail

PYTHON=/opt/miniconda3/envs/pmlr/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results/may_07_512_encoder"
EMBED_DIM=512
MODEL_SIZE="twin"

# Choose dataset: --local (1y) is faster, --large-local (5y) is more
# representative. Override via first arg, e.g. ./run_pool_ablation_local.sh --large-local
DATA_FLAG="${1:---local}"
case "$DATA_FLAG" in
    --local|--large-local) ;;
    *) echo "First arg must be --local or --large-local (got: $DATA_FLAG)"; exit 1 ;;
esac

# Sanity checks
for f in \
    "best_mae_model_${MODEL_SIZE}_d${EMBED_DIM}.pth" \
    "best_ijepa_model_${MODEL_SIZE}_d${EMBED_DIM}.pth" \
    "data_mean.npy" \
    "data_std.npy"; do
    if [ ! -f "$RESULTS_DIR/$f" ]; then
        echo "ERROR: missing $RESULTS_DIR/$f"; exit 1
    fi
done

cd "$REPO_ROOT"

# Sample sizes — kept modest because eval runs on CPU locally. Bump if you have time.
N_PROBE_SAMPLES=500
N_SEVERITY_STEPS=8
N_FORECAST_SAMPLES=200
BATCH_SIZE=32
# num_workers=0 avoids "netCDF4 Dataset not picklable" errors when DataLoader
# spawn workers try to serialize the dataset. CPU eval doesn't benefit from
# workers anyway since --local data is eagerly loaded into memory.
NUM_WORKERS=0

run_eval_pass() {
    local POOL_MODE="$1"
    echo
    echo "============================================================"
    echo "Pooling mode: $POOL_MODE   |   Data: $DATA_FLAG"
    echo "============================================================"
    export EXTRACT_FEATURES_POOLING="$POOL_MODE"

    echo -e "\n--- Phase 3: probe regression ($POOL_MODE) ---"
    "$PYTHON" eval/eval_probe.py \
        --model both \
        --mae-size "$MODEL_SIZE" --ijepa-size "$MODEL_SIZE" \
        --mae-embed-dim "$EMBED_DIM" --ijepa-embed-dim "$EMBED_DIM" \
        --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
        --n-probe-samples "$N_PROBE_SAMPLES" \
        --output-dir "$RESULTS_DIR" \
        "$DATA_FLAG"

    echo -e "\n--- Phase 4: FID/MMD vs corruption severity ($POOL_MODE) ---"
    "$PYTHON" eval/eval_distances.py \
        --model both \
        --mae-size "$MODEL_SIZE" --ijepa-size "$MODEL_SIZE" \
        --mae-embed-dim "$EMBED_DIM" --ijepa-embed-dim "$EMBED_DIM" \
        --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
        --n-severity-steps "$N_SEVERITY_STEPS" \
        --output-dir "$RESULTS_DIR" \
        "$DATA_FLAG"

    echo -e "\n--- Phase 5: ERA5 vs Pangu/GraphCast ($POOL_MODE) ---"
    "$PYTHON" eval/eval_real_vs_forecast.py \
        --model-size "$MODEL_SIZE" \
        --embed-dim "$EMBED_DIM" \
        --n-samples "$N_FORECAST_SAMPLES" \
        --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
        --output-dir "$RESULTS_DIR" \
        --local
}

run_eval_pass mean
run_eval_pass max

echo
echo "============================================================"
echo "Done. Compare side-by-side in:"
echo "  $RESULTS_DIR/plots/"
echo "Filenames with _pool-mean vs _pool-max are the ablation pair."
echo "============================================================"
