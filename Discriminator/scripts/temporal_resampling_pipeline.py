"""Fold-aware orchestration and aggregation for the baseline pipeline."""

from __future__ import annotations

import csv
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

try:
    from .temporal_resampling import (
        active_schedule, aggregate_draws, fixed_schedules, learned_schedules,
        settings, write_csv_gz, write_manifest,
    )
except ImportError:
    from temporal_resampling import (
        active_schedule, aggregate_draws, fixed_schedules, learned_schedules,
        settings, write_csv_gz, write_manifest,
    )


STANDARD_KEYS = {
    "lead_time": ["label", "variable", "lead_hour", "is_null"],
    "corruption_strength": ["label", "variable", "corruption", "severity", "is_null"],
}
DISCRIMINATOR_KEYS = [
    "architecture", "input_variables", "encoder_pretraining", "kind", "target",
    "x", "source", "is_era5_test_null",
]


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return path


def _ranges(cfg):
    monthly = cfg.monthly_split
    return [monthly.corruption_time_range, *list(monthly.model_valid_time_ranges)]


def _child_id(parent_id, schedule):
    return f"{parent_id}--{schedule.resample_id}"


def _child_run_dir(cfg, schedule):
    return Path(str(cfg.pipeline.run_dir)) / "resamples" / schedule.family / _child_id(cfg.pipeline.id, schedule)


def _child_output_root(cfg, schedule):
    try:
        from .plot_standard_metric_baselines import baseline_output_dir, variables_from_config
    except ImportError:
        from plot_standard_metric_baselines import baseline_output_dir, variables_from_config
    child = _child_cfg(cfg, schedule, [])
    run_dir = _child_run_dir(cfg, schedule)
    child.baseline.output_dir = str(run_dir)
    return baseline_output_dir(child, variables_from_config(child))


def _prior_checkpoint(cfg, schedule):
    configured = settings(cfg).get("input_run_dir") or cfg.pipeline.get("input_checkpoint_dir")
    if configured is None:
        return None
    root = Path(str(configured))
    if root.name == "target_discriminators":
        return root
    candidates = sorted(root.glob(
        f"resamples/learned/*--{schedule.resample_id}/*/models/target_discriminators"
    ))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one checkpoint directory for {schedule.resample_id} under {root}; found {candidates}"
        )
    return candidates[0]


def _child_cfg(cfg, schedule, stages, checkpoint=None, resume=False):
    child = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    child.temporal_resampling.active_schedule = schedule.to_dict()
    child.pipeline.id = _child_id(str(cfg.pipeline.id), schedule)
    child.pipeline.runs_dir = str(Path(str(cfg.pipeline.run_dir)) / "resamples" / schedule.family)
    child.pipeline.stages = list(stages)
    child.pipeline.resume = bool(resume)
    child.pipeline.input_checkpoint_dir = None if checkpoint is None else str(checkpoint)
    return child


def _execute_children(cfg, schedules, stages, *, require_checkpoints=False):
    try:
        from .run_baseline_pipeline import execute_pipeline
    except ImportError:
        from run_baseline_pipeline import execute_pipeline
    manifests = []
    for schedule in schedules:
        checkpoint = None
        resume = bool(cfg.pipeline.get("resume", False))
        run_dir = _child_run_dir(cfg, schedule)
        if require_checkpoints:
            local = next(iter(sorted(run_dir.glob("*/models/target_discriminators"))), None)
            checkpoint = local if local is not None else _prior_checkpoint(cfg, schedule)
            if checkpoint is None:
                raise FileNotFoundError(f"No discriminator checkpoints found for {schedule.resample_id}")
            resume = run_dir.exists()
        child_stages = list(stages)
        if bool((cfg.get("histogram_matching", {}) or {}).get("enabled", False)):
            if "fit_histogram_matching" not in child_stages:
                child_stages.insert(0, "fit_histogram_matching")
        child = _child_cfg(cfg, schedule, child_stages, checkpoint=checkpoint, resume=resume)
        manifests.append(execute_pipeline(child))
    return manifests


def _validate_counts(rows, group_fields, count_fields):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)
    for key, group in grouped.items():
        for field in count_fields:
            values = {int(float(row[field])) for row in group if row.get(field, "") != ""}
            if len(values) > 1:
                raise ValueError(f"Inconsistent {field} across temporal resamples for {dict(zip(group_fields, key))}: {sorted(values)}")


def _metric_names(cfg):
    try:
        from .plot_standard_metric_baselines import metric_names_from_config
    except ImportError:
        from plot_standard_metric_baselines import metric_names_from_config
    return metric_names_from_config(cfg)


def aggregate_standard(cfg, schedules):
    metric_names = _metric_names(cfg)
    root = Path(str(cfg.pipeline.run_dir))
    output_root = _parent_output_root(cfg)
    draw_rows = []
    outputs = []
    for experiment, keys in STANDARD_KEYS.items():
        source_name = f"{experiment}.csv"
        rows = []
        for schedule in schedules:
            path = _child_output_root(cfg, schedule) / "data" / source_name
            for source in _read_csv(path):
                source["resample_id"] = schedule.resample_id
                rows.append(source)
                for metric in metric_names:
                    draw_rows.append({
                        "family": "fixed", "resample_id": schedule.resample_id,
                        "experiment": experiment, "metric": metric,
                        **{key: source.get(key, "") for key in keys},
                        "value": source[metric], "n_samples": source.get("n_samples", ""),
                        "pairwise_n_samples": source.get("pairwise_n_samples", ""),
                    })
        _validate_counts(rows, keys, ("n_samples", "pairwise_n_samples"))
        aggregate = aggregate_draws(rows, metric_names, keys, "p05_p95")
        # Non-metric metadata are constant by construction; retain representative values.
        by_key = {tuple(row.get(key) for key in keys): row for row in rows}
        for row in aggregate:
            representative = by_key[tuple(row.get(key) for key in keys)]
            for field in ("n_samples", "pairwise_n_samples", "n_pairs", "initialization_start",
                          "initialization_end", "valid_start", "valid_end"):
                if field in representative:
                    row[field] = representative[field]
        outputs.append(_write_csv(output_root / "data" / source_name, aggregate))
    outputs.append(_write_csv(output_root / "data" / "fixed_metric_draws.csv", draw_rows))
    split_path = aggregate_split_manifests(cfg, schedules, "fixed")
    if split_path is not None:
        outputs.append(split_path)
    canonical = schedules[min(4, len(schedules) - 1)]
    canonical_data = _child_output_root(cfg, canonical) / "data"
    for name in ("scwd_anchor_contributions.nc", "global_mean_wasserstein_distributions.nc",
                 "corruption_disturbances.nc"):
        source = canonical_data / name
        if source.is_file():
            target = output_root / "data" / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target); outputs.append(target)
    return outputs


def aggregate_discriminator(cfg, schedules):
    output_root = _parent_output_root(cfg)
    rows, terms = [], []
    for schedule in schedules:
        data = _child_output_root(cfg, schedule) / "data"
        for row in _read_csv(data / "discriminator_reverse_kl.csv"):
            row["resample_id"] = schedule.resample_id; rows.append(row)
        terms_path = data / "discriminator_terms.csv.gz"
        if terms_path.is_file():
            with gzip.open(terms_path, "rt", newline="") as handle:
                for row in csv.DictReader(handle):
                    row["resample_id"] = schedule.resample_id; terms.append(row)
    _validate_counts(rows, DISCRIMINATOR_KEYS, ("n_samples", "ep_n_samples"))
    try:
        from .analyze_temporal_resamples import reconstruct_scores_from_terms
    except ImportError:
        from analyze_temporal_resamples import reconstruct_scores_from_terms
    temporary_terms = write_csv_gz(output_root / "data" / "discriminator_terms.csv.gz", terms)
    reconstructed = reconstruct_scores_from_terms(temporary_terms)
    for row in rows:
        key = (row["resample_id"], row["architecture"], row["kind"], row["target"],
               row["source"], float(row["x"]))
        if key not in reconstructed or not np.isclose(float(row["score"]), reconstructed[key], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Saved KL terms do not reconstruct draw {key}")
    aggregate = aggregate_draws(rows, ["score"], DISCRIMINATOR_KEYS, "minmax")
    representative = {tuple(row.get(key) for key in DISCRIMINATOR_KEYS): row for row in rows}
    for row in aggregate:
        source = representative[tuple(row.get(key) for key in DISCRIMINATOR_KEYS)]
        for field in ("stderr", "n_samples", "ep_train", "ep_n_samples"):
            row[field] = source.get(field, "")
        matching = [item for item in rows if tuple(item.get(key) for key in DISCRIMINATOR_KEYS) == tuple(row.get(key) for key in DISCRIMINATOR_KEYS)]
        row["checkpoint_paths"] = ",".join(item.get("checkpoint_path", "") for item in matching)
        row["checkpoint_sha256s"] = ",".join(item.get("checkpoint_sha256", "") for item in matching)
    score_path = _write_csv(output_root / "data" / "discriminator_reverse_kl.csv", aggregate)
    terms_path = temporary_terms
    draw_path = _write_csv(output_root / "data" / "discriminator_metric_draws.csv", rows)
    split_path = aggregate_split_manifests(cfg, schedules, "learned")
    return [score_path, terms_path, draw_path] + ([split_path] if split_path is not None else [])


def _parent_output_root(cfg):
    try:
        from .plot_standard_metric_baselines import baseline_output_dir, variables_from_config
    except ImportError:
        from plot_standard_metric_baselines import baseline_output_dir, variables_from_config
    return baseline_output_dir(cfg, variables_from_config(cfg))


def aggregate_split_manifests(cfg, schedules, family):
    output_root = _parent_output_root(cfg)
    rows = []
    for schedule in schedules:
        path = _child_output_root(cfg, schedule) / "data" / "split_manifest.csv.gz"
        if not path.is_file():
            continue
        with gzip.open(path, "rt", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return write_csv_gz(output_root / "data" / f"{family}_split_manifest.csv.gz", rows) if rows else None


def aggregate_all_draws(cfg):
    output_root = _parent_output_root(cfg)
    sources = [output_root / "data" / "fixed_metric_draws.csv",
               output_root / "data" / "discriminator_metric_draws.csv"]
    rows = []
    for path in sources:
        if path.is_file():
            for row in _read_csv(path):
                row["source_table"] = path.name; rows.append(row)
    draw_path = _write_csv(output_root / "data" / "metric_draws.csv", rows) if rows else None
    split_rows = []
    for path in (output_root / "data" / "fixed_split_manifest.csv.gz",
                 output_root / "data" / "learned_split_manifest.csv.gz"):
        if path.is_file():
            with gzip.open(path, "rt", newline="") as handle:
                split_rows.extend(csv.DictReader(handle))
    if split_rows:
        write_csv_gz(output_root / "data" / "split_manifest.csv.gz", split_rows)
    return draw_path


def run_resampled_stage(stage, cfg, tracker, output_root, resolved_path):
    """Run one parent stage over its required schedules."""
    learned = learned_schedules(cfg)
    fixed = fixed_schedules(cfg, _ranges(cfg))
    if stage == "plot":
        canonical = learned[min(4, len(learned) - 1)]
        checkpoint = next(iter(sorted(_child_run_dir(cfg, canonical).glob("*/models/target_discriminators"))), None)
        if checkpoint is None:
            checkpoint = _prior_checkpoint(cfg, canonical)
        if checkpoint is not None:
            OmegaConf.update(cfg, "target_discriminator.checkpoint_dir", str(checkpoint), merge=False)
            if cfg.baseline.get("discriminator") is not None:
                OmegaConf.update(cfg, "baseline.discriminator.checkpoint_dir", str(checkpoint), merge=False)
        if bool((cfg.get("histogram_matching", {}) or {}).get("enabled", False)):
            source_maps = _child_run_dir(cfg, canonical) / "data" / "histogram_matching"
            target_maps = Path(str(cfg.baseline.output_dir)) / "data" / "histogram_matching"
            if source_maps.is_dir():
                shutil.copytree(source_maps, target_maps, dirs_exist_ok=True)
        return None
    if stage == "fit_histogram_matching":
        # Fitting is performed inside each consuming child so maps are train-fold specific.
        return [], []
    if stage == "train_discriminators":
        manifests = _execute_children(cfg, learned, ["train_discriminators"])
        return [m["manifest_path"] for m in manifests], manifests
    if stage == "evaluate_standard_metrics":
        manifests = _execute_children(cfg, fixed, ["evaluate_standard_metrics"])
        paths = aggregate_standard(cfg, fixed); draw_path = aggregate_all_draws(cfg)
        if draw_path is not None:
            paths.append(draw_path)
        with tracker.run("evaluation/standard-metrics-aggregate", "temporal-resampling-aggregate", cfg,
                         tags=["evaluation", "standard-metrics", "temporal-resampling"]) as run:
            tracker.log_csv_table(run, "metrics/temporal_standard_draws",
                                  _parent_output_root(cfg) / "data" / "fixed_metric_draws.csv")
            if bool(cfg.pipeline.wandb.get("upload_evaluation_data", True)):
                tracker.log_artifact(run, "temporal-standard-metric-samples", "evaluation",
                                     [*paths, resolved_path], metadata={"pipeline_id": tracker.group})
        return [str(path) for path in paths], manifests
    if stage == "evaluate_discriminator_metrics":
        manifests = _execute_children(
            cfg, learned, ["evaluate_discriminator_metrics"], require_checkpoints=True,
        )
        paths = aggregate_discriminator(cfg, learned); draw_path = aggregate_all_draws(cfg)
        if draw_path is not None:
            paths.append(draw_path)
        with tracker.run("evaluation/discriminator-metrics-aggregate", "temporal-resampling-aggregate", cfg,
                         tags=["evaluation", "discriminator-metrics", "temporal-resampling"]) as run:
            tracker.log_csv_table(run, "metrics/temporal_discriminator_draws",
                                  _parent_output_root(cfg) / "data" / "discriminator_metric_draws.csv")
            if bool(cfg.pipeline.wandb.get("upload_evaluation_data", True)):
                tracker.log_artifact(run, "temporal-discriminator-metric-samples", "evaluation",
                                     [*paths, resolved_path], metadata={"pipeline_id": tracker.group})
        return [str(path) for path in paths], manifests
    return None


def finalize_resampling_manifest(cfg):
    output_root = _parent_output_root(cfg)
    data = output_root / "data"
    paths = list(data.glob("*")) if data.exists() else []
    payload = {
        "pipeline_id": str(cfg.pipeline.id),
        "learned_schedules": [item.to_dict() for item in learned_schedules(cfg)],
        "fixed_schedules": [item.to_dict() for item in fixed_schedules(cfg, _ranges(cfg))],
        "learned_bounds": "minmax", "fixed_bounds": "p05_p95",
        "within_test_bootstrap": False,
    }
    return write_manifest(data / "resampling", payload, paths)
