#!/bin/bash
# Local smoke test for 768-dim MAE pooling comparison.
#
# Test 1: embed_dim=384 + concat pooling (mean+max) → 768 effective latent dim
# Test 2: embed_dim=768 + max pooling              → 768 latent dim
#
# Trains for 2 epochs locally (just to verify the pipeline runs end-to-end).
# For a real comparison, run submit_pool768_comparison.sh on the cluster.

set -eo pipefail

PYTHON=/opt/miniconda3/envs/pmlr/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DIR_CONCAT="results/pool768_concat384"
DIR_MAX="results/pool768_max768"
mkdir -p "$DIR_CONCAT" "$DIR_MAX"

# ── Test 1: embed_dim=384 + concat pooling ───────────────────────────────────
echo
echo "============================================================"
echo "Test 1: embed_dim=384 + concat pooling (384+384=768)"
echo "============================================================"

echo "--- Training MAE embed_dim=384 (2 epochs local) ---"
"$PYTHON" train/train_mae.py \
    --local \
    --embed-dim 384 \
    --output-dir "$DIR_CONCAT"

echo "--- Evaluating with concat pooling ---"
"$PYTHON" eval/eval_real_vs_forecast.py \
    --local \
    --mae-only \
    --pooling concat \
    --embed-dim 384 \
    --n-samples 50 \
    --batch-size 16 \
    --num-workers 0 \
    --output-dir "$DIR_CONCAT"

# ── Test 2: embed_dim=768 + max pooling ─────────────────────────────────────
echo
echo "============================================================"
echo "Test 2: embed_dim=768 + max pooling"
echo "============================================================"

echo "--- Training MAE embed_dim=768 (2 epochs local) ---"
"$PYTHON" train/train_mae.py \
    --local \
    --embed-dim 768 \
    --output-dir "$DIR_MAX"

echo "--- Evaluating with max pooling ---"
"$PYTHON" eval/eval_real_vs_forecast.py \
    --local \
    --mae-only \
    --pooling max \
    --embed-dim 768 \
    --n-samples 50 \
    --batch-size 16 \
    --num-workers 0 \
    --output-dir "$DIR_MAX"

echo
echo "============================================================"
echo "Smoke test done. Results in:"
echo "  $DIR_CONCAT/plots/real_vs_forecast/"
echo "  $DIR_MAX/plots/real_vs_forecast/"
echo "============================================================"
