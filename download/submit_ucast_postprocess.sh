#!/bin/bash
#SBATCH --job-name=ucast_post
#SBATCH --account=pmlr_jobs
#SBATCH --partition=jobs
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=ucast_post_%j.out
#SBATCH --error=ucast_post_%j.err

# Post-processing for the 2020 U-Cast run. Submit with a dependency so it starts only
# if the inference job actually succeeded:
#
#     sbatch --dependency=afterok:<inference_jobid> download/submit_ucast_postprocess.sh
#
# The QOS here is MaxJobsPU=1 / MaxSubmitJobsPU=3, so this can sit PENDING while the
# inference job runs; only three jobs may be submitted at once.
#
# This deliberately stops short of running the discriminator pipeline. It prepares the
# inputs that pipeline needs and validates that its config resolves with U-Cast
# registered -- cheap and safe. Training critics is expensive, and launching it blind
# behind a multi-hour job risks discovering a config typo after burning that time.

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pmlr

REPO=~/atmospheric-fields
FINAL=/work/scratch/ddemler/ucast2020_final
MEMBERS=/work/scratch/ddemler/ucast2020_members
PLOTS=/work/scratch/ddemler/ucast2020_plots

echo "node: $(hostname)"
mkdir -p "$MEMBERS" "$PLOTS"

echo "───── 1. inference output ─────"
ls -la "$FINAL" "$FINAL/nonsurf" 2>/dev/null

echo "───── 2. verifying the converted files ─────"
python - <<'PY'
import glob, numpy as np, xarray as xr
for path in sorted(glob.glob("/work/scratch/ddemler/ucast2020_final/**/*.nc", recursive=True)):
    ds = xr.open_dataset(path)
    var = list(ds.data_vars)[0]
    n_nan = int(np.isnan(ds[var].values[:2]).sum())   # sample, not the whole 27 GB
    print(f"  {path.split('/')[-1]}")
    print(f"     vars={list(ds.data_vars)}")
    print(f"     sizes={dict(ds.sizes)}  NaN(sample)={n_nan}")
PY

echo "───── 3. splitting the ensemble into per-member files ─────"
# The discriminator pipeline consumes deterministic files, one field per
# (time, prediction_timedelta). Members become separate entries in its model list.
python "$REPO/ucast/extract_members.py" \
  --inputs "$FINAL"/ucast_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc \
           "$FINAL"/nonsurf/ucast_6steps_1.5deg_2020-01-01_2020-12-31.nc \
  --output-dir "$MEMBERS" \
  --members 0 1 2 3 4

echo "───── 4. checking the pipeline config resolves with U-Cast registered ─────"
# --cfg job resolves and prints the config without running any stage, so a bad override
# fails here in seconds instead of after hours of discriminator training.
cd "$REPO/Discriminator"
python scripts/run_baseline_pipeline.py --cfg job \
  --config-path conf --config-name baseline_pipeline \
  "output_dir=$PLOTS" \
  "pipeline.id=ucast2020" \
  "+baseline.forecast_files.U-Cast-m0=[$MEMBERS/ucast-m0_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc]" \
  2>&1 | grep -A6 -E "forecast_files|output_dir" | head -30 \
  || echo "  CONFIG RESOLUTION FAILED -- fix the override before running the pipeline"

echo
echo "POSTPROCESS OK"
echo "Per-member files: $MEMBERS"
echo "Plots will go to: $PLOTS  (scratch is writable from compute nodes; team07 is not)"
echo
echo "To copy the dataset into team07, run this from a LOGIN node:"
echo "  cp -r $FINAL/* /cluster/courses/pmlr/teams/team07/data/"
