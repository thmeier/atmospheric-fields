# Handoff — temporal-resampling paper run (2026-08-25/26)

Context for a fresh session picking up the corruption blind-spot / reverse-KL
figures. Branch **`ucast-eval`** (`git@github.com:thmeier/atmospheric-fields`),
everything below is pushed. Read `memory/MEMORY.md` and
`~/.claude/projects/-Users-dimstademler-Desktop-eth-local-PMLR-L-atmospheric-fields/memory/`
too — cluster facts live there.

---

## 1. What the task was

Reproduce four figures plus the blind-spot N/M table, over Younes's new
temporal-resampling protocol (per-month random 7-day test windows with a
buffer) instead of the old single fixed days 20–26 window.

Deliverables, all produced:

| Figure | Path under `<run>/<variable-tag>/plots/paper/` |
|---|---|
| Normalized metrics vs corruption severity | `corruption_by_type_normalized.png` |
| Reverse-KL vs corruption strength | `discriminator/squeezenet/corruption_strength_reverse_kl.png` |
| Reverse-KL vs lead time | `discriminator/squeezenet/lead_time_reverse_kl.png` |
| Real-vs-fake logits per lead | `target_logit_distributions/squeezenet/forecast/<model>/all_lead_times.png` |
| Blind-spot table | `<run>/blindspots/blindspot_table.tex` |

"Ours" = the SFNO/embedding metric and is **excluded** (`sfno.enabled=false`);
squeezenet critics ARE trained.

---

## 2. Current state

- **Completed run: `paper-temporal-106911`** (job 106950 finished it).
  Config: **T2M only**, `fixed_replicates=25`, `learned_replicates=5`,
  9 corruptions, 7 forecast models, paper profile.
- Results: `/cluster/courses/pmlr/teams/team07/results/paper-temporal-106911`
  (721 MB, 85 critics, all raw draws + NetCDFs).
- Local copy: `figures/paper-temporal-106911/` (407 MB, PNG+PDF+NPZ + CSVs).
- **Cluster queue is empty.** Nothing running or queued.
- The user intends to **change things and rerun everything**, so treat the
  existing run as a reference to diff against, not as final.

### Submit command

```bash
sbatch Discriminator/scripts/submit_paper_temporal_pipeline.sh \
  "baseline.variables=[2m_temperature]" temporal_resampling.fixed_replicates=25
# resume (skips completed folds/resamples):
PIPELINE_ID=<id> RESUME=true sbatch Discriminator/scripts/submit_paper_temporal_pipeline.sh ...
```

Helpers installed on the cluster: `~/paper_status.sh [jobid]`,
`~/backup_checkpoints.sh` (scratch → team07, incremental, safe mid-job).

---

## 3. Cluster facts that cost time to learn

- **Every job is clamped to 1 GPU / 3 CPUs.** All CPU request forms are
  *rejected*: `--cpus-per-task`, `-c`, `--cpus-per-gpu`, `--gres` all error.
  `--gpus=8 --constraint=2080ti` is accepted but still allocates
  `cpu=3,gres/gpu=1`. Size pools from `nproc`, never from an intended
  allocation. Whole-node parallelism is **not obtainable**.
  Use `sbatch --test-only` to validate flags for free.
- **QOS**: MaxJobsPU=1 (one *runs*), MaxSubmitJobsPU=3 (three may be queued).
  `--dependency=afterany:<id>` works. Walltime caps at 24h.
- **The `pmlr` conda env has no hydra.** Run via
  `/work/scratch/ddemler/embedding_smoke/venv/bin/python` with
  `PROJ_DATA=$HOME/miniconda3/envs/pmlr/share/proj` (cartopy).
- **Storage**: `$HOME` is a 20 GB quota, ~17 GB used — never write results
  there. Work on `/work/scratch/ddemler` (100 GB), keep durable output in
  `/cluster/courses/pmlr/teams/team07/results` (huge, team-readable via ACLs).
  `CLAUDE.md:25` claims scratch is auto-cleaned every 1–7 days; **the user says
  it is not cleaned that often** — don't use that line to justify writing to home.
- `--constraint=2080ti`, not 5060ti: the latter is sm_120 and needs
  torch ≥2.7/cu128; the env ships torch 2.5.1+cu121.

---

## 4. Bugs found and fixed (all pushed)

| Commit | Problem |
|---|---|
| `f5beae0` | Merged Younes's `discriminator` branch; his side was a strict superset for the conflicted files |
| `3276aa7` | Local/remote `ucast-eval` had diverged into duplicate rebased history — merged, **never force-pushed** |
| `cd3baa5` | Added process pool over resamples + `blindspots_from_temporal_draws.py` |
| `7e1aa8e` | sbatch flags didn't match cluster limits |
| `858fe6b` | Used the scratch venv + scratch output dir |
| `39fb174` | **Resumable resample children** + attribution gated to the canonical fold |
| `f14c541` | **SWIFT NetCDF was pathologically chunked** (see below) |
| `44aaf92`, `9c25f9f` | Keep checkpoints; move durable output to team07 |
| `e362cad` | `_validate_counts` fatally rejected legitimate varying sample counts |
| `7bba6e3` | Canonical training plots were looked for at the wrong path |
| `2572ef9` | Panel titles overlapped axis labels at paper profile |
| `4f0a4a6` | **Exclude non-converged critics** |

### SWIFT chunking (big one)

`swift_*.nc` was written with `zlib=True, chunks=[244,2,41,80]`, so reading one
`(time, lead)` field decompresses chunks spanning 244 timesteps. Training ran at
**7.8 s/step vs GraphCast's 0.16 s/step (~49×)**. A contiguous uncompressed
rewrite is byte-identical and reads **154× faster** (123 ms → 0.8 ms per field).
The submit script rebuilds it on scratch automatically; the shared team file is
deliberately untouched. `FeatureMetric/scripts/convert_swift_zarr_to_netcdf.py`
now defaults to uncompressed and chunks per-timestep when compression is asked for.

### Non-converged critic (affects Fig 3 left)

One GraphCast critic in `learned_01` hit **0.888 train / 0.521 test accuracy
(chance)** and scored **−230** where its four siblings scored +7…+14, dragging
the mean to −37. **Median does not fix this** — the learned band is a min–max
envelope over 5 folds, and with n=5 even p05–p95 interpolates between the first
two order statistics. The draw must leave the sample.
`temporal_resampling.min_critic_test_accuracy` (default 0.6) now drops it,
reads held-out accuracy only (never the score), and logs to
`data/excluded_critics.csv`. Effect: 192h GraphCast goes
`−37.3 [−230.6, 13.7]` → `+11.0 [6.9, 13.7]`.

**The completed run predates this fix**, so its `lead_time_reverse_kl.png` is
still wrong. Re-running just the plot stage on a resume would regenerate it.

---

## 5. Measured performance (use these, not guesses)

- Old reference run, 1 variable, 1 split: `evaluate_standard_metrics` **34.7 min**,
  `evaluate_discriminator_metrics` **46.4 min**.
- **4 variables cost 2.61×** the standard-metrics stage vs 1 variable
  (measured with the full 10-metric set; a 3-cheap-metric test misleadingly
  suggests 1.63× because SCWD/MMD dominate at scale).
- Per resample at T2M: **~15 min**. Training: **~4.7h** for 85 critics
  (forecast targets 140 steps ≈ 2 min; corruption targets 880 steps ≈ 5 min).
- Completed T2M/25-replicate run ≈ **15.5h** total.
  A 4-variable/50-replicate run was projected at **~44h** (needs 2–3 job windows).
- The process pool only buys ~2–3× because of the 3-CPU clamp.

**Resume is cheap and works**: re-running after the aggregation bug skipped all
5 training folds and all 25 resamples and finished in **30 min** (standard
metrics 0.7 min, discriminator eval 25.2 min, plot 4.2 min).

---

## 6. Scientific findings

### Corruption results reproduce the published table

**N (detection): 25/27 compared cells agree.** Both disagreements were
already predicted:

| Cell | Published | New | Margin |
|---|---|---|---|
| `grf × Zonal spec.` | ✗ | **✓** | 2.67× threshold |
| `hf_noise × Raw MMD` | ✓ | **✗** | 0.97× threshold |

`grf × Zonal spec.` is the cell `memory/blindspot-table-null-threshold.md`
already flags as needing recheck (it saw 3.64×). `hf_noise × Raw MMD` is the
predicted flip from replacing the single null with a resampled one.

**Caveat: 27 = 9 corruptions × 3 columns.** The published table has 4 metric
columns; **the SCWD column was untested** because the generator keys on
`scwd_area_weighted` and the run only computed plain `scwd`. That's now added to
`baseline_config.yaml` metrics, but costs extra compute and has not been run.

**M (monotonic): only 17/27 agree**, all in one direction (published ✗, mine ✓).
`M` uses non-strict `b >= a`, so a flat curve carrying no signal passes. Every
mismatch has margin 0.00–0.70×, i.e. metrics that don't detect the corruption at
all. **Decide whether the paper means strictly increasing.**

### Buffer / window protocol

Windows start on days 5–20 (`random_test_start_range`), so the ±4-day buffer
always fits inside a month — the user endorsed this over "any 7 consecutive days".
Verified over 11,400 (draw, month) pairs: the leading buffer never crosses
backwards; **114 pairs, all Februaries, spill 1–2 days into March** (a window
ending day 26 wants buffer through day 30). Conservative, not a correctness bug.

### Sample-count variation

`lead_time` forecast rows legitimately differ per fold (154–168 pairs) because a
forecast initialised near a window's end has its +192h valid time outside it.
Corruption rows are all identical at n=1000 (the cap binds). This is recorded as
`n_samples_min`/`n_samples_max`, not treated as an error. The MMD O(1/n) bias
this implies is **0.47% of the p05–p95 spread at worst — negligible**, and
affects no target figure.

### Reproducibility

`mmd_rbf` shifts up to **2e-3 relative when the BLAS thread count changes**
(bandwidth is a median heuristic). `mean_bias` and `global_mean_wasserstein` are
bit-identical. The pool pins threads per worker, which makes runs *more*
reproducible: two pooled runs agreed 1080/1080 draws, and a sequential run
pinned to the same thread count reproduced them exactly.

---

## 7. Open items

1. **Regenerate Fig 3 left** — the completed run predates the critic-exclusion
   fix. Only the plot stage needs re-running.
2. **SCWD column** — `scwd_area_weighted` is now in the metrics list but has
   never been computed. Needs a full resample stage (~7h at T2M/25).
3. **Monotonicity definition** — strict vs non-strict; 10 table cells hinge on it.
4. **`target_discriminator.output_dir` is pinned to a hard-coded four-variable
   tag** in `conf/target_discriminator_baselines.yaml` regardless of
   `baseline.variables`. This is why single-variable training output landed in a
   `2m_temperature__10m_u...msl/` directory and silently broke the plot copy.
   Worth making it follow `${baseline.variables}`.
5. **`CLAUDE.md:25`** still states scratch is auto-cleaned every 1–7 days; the
   user disputes this. Reconcile.
6. `memory/` is gitignored (`.gitignore:43`) although `CLAUDE.md` describes it
   as committed.

---

## 8. Working conventions the user asked for

- **Never** add `Co-Authored-By: Claude`, "Generated with Claude Code", or any
  AI attribution to commits or PR bodies. (Audited: the pushed branch is clean.)
- Commit subjects **shorter than seven words**.
- Plan non-trivial work before implementing; surface design choices rather than
  deciding alone.
- Don't force-push `ucast-eval` — it is shared with Younes and thmeier.
