"""Check that a bootstrap blind-spot figure bundle can be replotted from its NPZ.

`save_figure_bundle` captures the rendered artists, which alone would only give
back the mean line and the band edges that happened to be drawn. The figure also
ships the raw per-replicate scores as `input_*` arrays so the band definition can
be changed later. This verifies that promise: every drawn curve is reconstructed
from the raw payload, and a few alternative band definitions are computed from it.

    python scripts/verify_bootstrap_bundle.py \
        <path>/corruption_by_type_bootstrap_null.npz
"""
import sys, numpy as np, json
path = sys.argv[1]
z = np.load(path, allow_pickle=False)
meta = json.loads(str(z["metadata_json"]))
print("plot_type:", meta["plot_type"])
print("payload keys:", sorted(k for k in z.files if k.startswith("input_")))

scores = z["input_curve_scores"]
draws = z["input_null_draws"]
scales = z["input_normalization_scales"]
corruptions = [str(x) for x in z["input_corruptions"]]
metrics = [str(x) for x in z["input_metrics"]]
severities = z["input_curve_severities"]
print(f"curve_scores {scores.shape}  null_draws {draws.shape}  "
      f"corruptions={len(corruptions)}  metrics={len(metrics)}")

# errorbar() injects its own Line2D artists, so line index does not track metric
# index. Match each reconstructed curve against every line drawn on its panel.
worst, checked = 0.0, 0
for c, corruption in enumerate(corruptions):
    drawn_lines = [z[k] for k in sorted(z.files)
                   if k.startswith(f"axes_{c}_line_") and k.endswith("_y")]
    valid = np.isfinite(severities[c])
    for m, metric in enumerate(metrics):
        expected = np.nanmean(scores[c, valid, :, m], axis=1) / scales[m]
        best = min(
            (float(np.nanmax(np.abs(line - expected))) for line in drawn_lines
             if line.shape == expected.shape),
            default=np.inf,
        )
        worst = max(worst, best)
        checked += 1
print(f"\nreproduced {checked} drawn mean curves from the raw payload; max abs diff = {worst:.3e}")
assert worst < 1e-9, "payload does not reproduce the rendered curves"

m = metrics.index("mmd_rbf")
print(f"\nalternative bands for the mmd_rbf null ({draws.shape[0]} draws), unnormalized:")
for label, lo, hi in [("+/-1 sd", None, None), ("p5-p95", 5, 95), ("IQR", 25, 75), ("p1-p99", 1, 99)]:
    if lo is None:
        centre, half = draws[:, m].mean(), draws[:, m].std(ddof=1)
        print(f"  {label:8s} {centre - half:.6f} .. {centre + half:.6f}")
    else:
        print(f"  {label:8s} {np.percentile(draws[:, m], lo):.6f} .. {np.percentile(draws[:, m], hi):.6f}")
print("\nOK: bands can be redefined from the NPZ alone.")
