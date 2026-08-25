"""Run selectable baseline stages with grouped W&B tracking."""

import csv
import json
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

try:
    from .baseline_pipeline_tracking import PipelineTracker, safe_name
    from .plot_bundles import all_plot_bundle_paths, configure_plot_bundle_saving_from_cfg
    from .plot_standard_metric_baselines import (
        baseline_get,
        baseline_output_dir,
        evaluate_discriminator_metrics,
        evaluate_mmd_global_moment_matching,
        evaluate_standard_metrics,
        metric_names_from_config,
        mmd_global_moment_matching_output_dir,
        plot_mmd_global_moment_matching,
        plot_saved_standard_metric_baselines,
        read_mmd_global_moment_matching,
        variables_from_config,
    )
    from .train_target_discriminator_baselines import (
        plot_target_discriminator_interpretability, train_target_discriminator_baselines,
    )
    from .fit_histogram_matching import fit_histogram_matching_maps
    from .evaluate_bootstrap_null import bootstrap_null_output_dir, evaluate_bootstrap_null
    from .plot_bootstrap_blindspots import plot_bootstrap_blindspots
except ImportError:
    from baseline_pipeline_tracking import PipelineTracker, safe_name
    from plot_bundles import all_plot_bundle_paths, configure_plot_bundle_saving_from_cfg
    from plot_standard_metric_baselines import (
        baseline_get,
        baseline_output_dir,
        evaluate_discriminator_metrics,
        evaluate_mmd_global_moment_matching,
        evaluate_standard_metrics,
        metric_names_from_config,
        mmd_global_moment_matching_output_dir,
        plot_mmd_global_moment_matching,
        plot_saved_standard_metric_baselines,
        read_mmd_global_moment_matching,
        variables_from_config,
    )
    from train_target_discriminator_baselines import (
        plot_target_discriminator_interpretability, train_target_discriminator_baselines,
    )
    from fit_histogram_matching import fit_histogram_matching_maps
    from evaluate_bootstrap_null import bootstrap_null_output_dir, evaluate_bootstrap_null
    from plot_bootstrap_blindspots import plot_bootstrap_blindspots


STAGES = (
    "fit_histogram_matching",
    "train_discriminators",
    "evaluate_standard_metrics",
    "evaluate_discriminator_metrics",
    "evaluate_mmd_global_moment_matching",
    "evaluate_bootstrap_null",
    "plot",
    "plot_mmd_global_moment_matching",
    "plot_bootstrap_blindspots",
)


def generated_pipeline_id():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"baseline-{timestamp}-{uuid.uuid4().hex[:8]}"


def pipeline_runs_dir(cfg):
    """Resolve the immutable parent that holds isolated pipeline-run directories."""
    configured = cfg.pipeline.get("runs_dir")
    # ``manifest_dir`` is retained as a compatibility fallback for older
    # experiment configs and minimal unit-test fixtures.
    return Path(str(configured if configured is not None else cfg.pipeline.manifest_dir))


def configure_run_output_dirs(cfg, pipeline_id, run_dir, runs_parent):
    """Point all baseline stages at one non-shared pipeline-run directory."""
    original_target_dir = Path(str(cfg.target_discriminator.output_dir))
    target_leaf = original_target_dir.name
    OmegaConf.update(cfg, "pipeline.id", pipeline_id, merge=False)
    OmegaConf.update(cfg, "pipeline.run_dir", str(run_dir), force_add=True)
    # Freeze this interpolation before changing baseline.output_dir so the
    # resolved per-run config still records the actual common run parent.
    OmegaConf.update(cfg, "pipeline.runs_dir", str(runs_parent), force_add=True)
    OmegaConf.update(cfg, "baseline.output_dir", str(run_dir), merge=False)
    # Some experiments deliberately override target_discriminator.output_dir
    # (e.g. an SFNO fine-tuning run). Preserve that final directory name while
    # moving it under this invocation's isolated root.
    target_output_dir = Path(run_dir) / target_leaf
    OmegaConf.update(cfg, "target_discriminator.output_dir", str(target_output_dir), merge=False)
    input_checkpoint_dir = cfg.pipeline.get("input_checkpoint_dir")
    checkpoint_dir = (
        Path(str(input_checkpoint_dir)) if input_checkpoint_dir is not None
        else target_output_dir / "models" / "target_discriminators"
    )
    OmegaConf.update(cfg, "target_discriminator.checkpoint_dir", str(checkpoint_dir), merge=False)
    if cfg.baseline.get("discriminator") is not None:
        OmegaConf.update(
            cfg,
            "baseline.discriminator.checkpoint_dir",
            str(checkpoint_dir),
            merge=False,
        )


def selected_stages(cfg):
    requested = [str(stage) for stage in cfg.pipeline.stages]
    unknown = sorted(set(requested) - set(STAGES))
    duplicates = sorted({stage for stage in requested if requested.count(stage) > 1})
    if unknown or duplicates:
        raise ValueError(f"Invalid pipeline stages; unknown={unknown}, duplicates={duplicates}")
    return [stage for stage in STAGES if stage in requested]


def require_files(paths, stage):
    missing = sorted({str(path) for path in paths if not Path(str(path)).is_file()})
    if missing:
        raise FileNotFoundError(
            f"Stage {stage} is missing required local input(s):\n" + "\n".join(missing)
        )


def csv_row_count(path):
    with open(path, newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def standard_input_paths(cfg):
    paths = [cfg.real_nc_file]
    for configured in baseline_get(cfg, "forecast_files", {}).values():
        paths.extend(str(path) for path in configured)
    return paths


def plot_input_paths(cfg, output_root):
    """Return only plot inputs produced by the selected standard-metric run.

    Target-discriminator results are optional: the plotter renders them when
    present, but standard-only evaluation must not require their CSV.
    """
    paths = [
        output_root / "data" / "lead_time.csv",
        output_root / "data" / "corruption_strength.csv",
    ]
    if "scwd" in metric_names_from_config(cfg):
        paths.append(output_root / "data" / "scwd_anchor_contributions.nc")
    return paths


def changed_plot_paths(output_root, before):
    paths = sorted((output_root / "plots").rglob("*.png"))
    return [path for path in paths if before.get(path) != path.stat().st_mtime_ns]


def plot_bundle_members(paths):
    """Expand rendered PNG paths to their sibling PDF and NPZ plot data."""
    members = []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".png":
            members.extend(member for member in all_plot_bundle_paths(path) if member.is_file())
        elif path.is_file():
            members.append(path)
    return list(dict.fromkeys(members))


def run_stage(stage, cfg, tracker, output_root, resolved_path):
    upload_data = bool(cfg.pipeline.wandb.get("upload_evaluation_data", True))
    if stage == "fit_histogram_matching":
        with tracker.run("preprocessing/histogram-matching", "histogram-matching", cfg, tags=["preprocessing", "histogram-matching"]) as run:
            paths = fit_histogram_matching_maps(cfg)
            if paths:
                summary = Path(str(cfg.baseline.output_dir)) / "data" / "histogram_matching" / "fit_summary.csv"
                tracker.log_csv_table(run, "histogram_matching/maps", summary)
                if upload_data:
                    tracker.log_artifact(run, "histogram-matching", "preprocessing", [*paths, resolved_path], metadata={"pipeline_id": tracker.group})
            run.summary["histogram_matching/enabled"] = bool((cfg.get("histogram_matching", {}) or {}).get("enabled", False))
            return [str(path) for path in paths], [{"run_url": getattr(run, "url", None)}]
    if stage == "train_discriminators":
        require_files([cfg.real_nc_file], stage)
        records = train_target_discriminator_baselines(cfg, tracker=tracker)
        summary_path = Path(str(cfg.target_discriminator.output_dir)) / "data" / "target_train_test_metrics.csv"
        cases_path = Path(str(cfg.target_discriminator.output_dir)) / "data" / "target_interpretability_cases.csv"
        with tracker.run("training/summary", "discriminator-training-summary", cfg, tags=["training", "summary"]) as run:
            tracker.log_csv_table(run, "metrics/target_train_test", summary_path)
            tracker.log_csv_table(run, "interpretability/cases", cases_path)
            if summary_path.is_file():
                run.summary["metrics/target_train_test_rows"] = csv_row_count(summary_path)
            if cases_path.is_file():
                run.summary["interpretability/case_rows"] = csv_row_count(cases_path)
                if upload_data:
                    tracker.log_artifact(run, "target-train-test-metrics", "evaluation", [summary_path, cases_path, resolved_path])
        paths = [record["path"] for record in records]
        paths.extend(record["interpretability_gallery"] for record in records
                     if record.get("interpretability_gallery"))
        paths.extend(record["sfno_representation_magnitude_gallery"] for record in records
                     if record.get("sfno_representation_magnitude_gallery"))
        if summary_path.is_file():
            paths.append(str(summary_path))
        if cases_path.is_file():
            paths.append(str(cases_path))
        return [str(path) for path in plot_bundle_members(paths)], records

    if stage == "evaluate_standard_metrics":
        require_files(standard_input_paths(cfg), stage)
        with tracker.run("evaluation/standard-metrics", "standard-metric-evaluation", cfg,
                         tags=["evaluation", "standard-metrics"]) as run:
            paths = evaluate_standard_metrics(cfg)
            for filename, key in (
                ("lead_time.csv", "metrics/lead_time"),
                ("corruption_strength.csv", "metrics/corruption_strength"),
            ):
                path = output_root / "data" / filename
                tracker.log_csv_table(run, key, path)
                if path.is_file():
                    run.summary[f"{key}_rows"] = csv_row_count(path)
            if upload_data:
                tracker.log_artifact(
                    run, "standard-metric-evaluation", "evaluation", [*paths, resolved_path],
                    metadata={"pipeline_id": tracker.group},
                )
            return [str(path) for path in plot_bundle_members(paths)], [{"run_url": getattr(run, "url", None)}]

    if stage == "evaluate_mmd_global_moment_matching":
        if bool((cfg.get("histogram_matching", {}) or {}).get("enabled", False)):
            raise ValueError("Histogram matching cannot be composed with the separate MMD global-moment diagnostic.")
        require_files(standard_input_paths(cfg), stage)
        with tracker.run("evaluation/mmd-global-moment-matching", "mmd-global-moment-matching", cfg,
                         tags=["evaluation", "mmd", "global-moment-matching"]) as run:
            paths = evaluate_mmd_global_moment_matching(cfg)
            mmd_root = mmd_global_moment_matching_output_dir(cfg, variables_from_config(cfg))
            csv_path = mmd_root / "data" / "mmd_global_moment_matching.csv"
            tracker.log_csv_table(run, "metrics/mmd_global_moment_matching", csv_path)
            if csv_path.is_file():
                run.summary["metrics/mmd_global_moment_matching_rows"] = csv_row_count(csv_path)
            if upload_data:
                tracker.log_artifact(
                    run, "mmd-global-moment-matching", "evaluation", [*paths, resolved_path],
                    metadata={"pipeline_id": tracker.group},
                )
            return [str(path) for path in plot_bundle_members(paths)], [{"run_url": getattr(run, "url", None)}]

    if stage == "plot_mmd_global_moment_matching":
        mmd_root = mmd_global_moment_matching_output_dir(cfg, variables_from_config(cfg))
        csv_path = mmd_root / "data" / "mmd_global_moment_matching.csv"
        require_files([csv_path], stage)
        with tracker.run("plotting/mmd-global-moment-matching", "mmd-global-moment-matching-plots", cfg,
                         tags=["plotting", "mmd", "global-moment-matching"]) as run:
            paths = plot_mmd_global_moment_matching(read_mmd_global_moment_matching(mmd_root), mmd_root)
            bundle_paths = plot_bundle_members(paths)
            tracker.log_images(run, paths, mmd_root / "plots")
            if bool(cfg.pipeline.wandb.get("upload_plots", True)):
                tracker.log_artifact(
                    run, "mmd-global-moment-matching-plots", "plots", [*bundle_paths, csv_path, resolved_path],
                    metadata={"pipeline_id": tracker.group},
                )
            return [str(path) for path in bundle_paths], [{"run_url": getattr(run, "url", None)}]

    if stage == "evaluate_bootstrap_null":
        require_files([cfg.real_nc_file], stage)
        with tracker.run("evaluation/bootstrap-null", "bootstrap-null-evaluation", cfg,
                         tags=["evaluation", "bootstrap-null", "blindspots"]) as run:
            paths = evaluate_bootstrap_null(cfg)
            data_root = bootstrap_null_output_dir(cfg, variables_from_config(cfg)) / "data"
            for filename, key in (
                ("bootstrap_null_draws.csv", "metrics/bootstrap_null_draws"),
                ("bootstrap_curves.csv", "metrics/bootstrap_curves"),
                ("bootstrap_null_verdicts.csv", "metrics/bootstrap_null_verdicts"),
            ):
                path = data_root / filename
                tracker.log_csv_table(run, key, path)
                if path.is_file():
                    run.summary[f"{key}_rows"] = csv_row_count(path)
            summary_path = data_root / "bootstrap_null_summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                for name, value in summary["thresholds"].items():
                    run.summary[f"bootstrap_null/threshold/{name}"] = value
                    run.summary[f"bootstrap_null/shipped/{name}"] = summary["shipped_single_null"][name]
                run.summary["bootstrap_null/replicates"] = summary["replicates"]
                run.summary["bootstrap_null/curve_replicates"] = summary["curve_replicates"]
            if upload_data:
                tracker.log_artifact(
                    run, "bootstrap-null-evaluation", "evaluation", [*paths, resolved_path],
                    metadata={"pipeline_id": tracker.group},
                )
            return [str(path) for path in paths], [{"run_url": getattr(run, "url", None)}]

    if stage == "plot_bootstrap_blindspots":
        data_root = bootstrap_null_output_dir(cfg, variables_from_config(cfg)) / "data"
        require_files([data_root / "bootstrap_curves.csv", data_root / "bootstrap_null_draws.csv",
                       data_root / "bootstrap_null_verdicts.csv",
                       data_root / "bootstrap_null_summary.json"], stage)
        with tracker.run("plotting/bootstrap-blindspots", "bootstrap-blindspot-plots", cfg,
                         tags=["plotting", "bootstrap-null", "blindspots"]) as run:
            paths = plot_bootstrap_blindspots(cfg)
            bundle_paths = plot_bundle_members(paths)
            table_paths = [
                data_root / name for name in
                ("blindspot_table.tex", "null_exceedance_table.tex", "null_exceedance.csv")
            ]
            tracker.log_images(run, paths, data_root.parent / "plots")
            tracker.log_csv_table(run, "metrics/null_exceedance", data_root / "null_exceedance.csv")
            if bool(cfg.pipeline.wandb.get("upload_plots", True)):
                tracker.log_artifact(
                    run, "bootstrap-blindspot-plots", "plots",
                    [*bundle_paths, *table_paths, resolved_path],
                    metadata={"pipeline_id": tracker.group},
                )
            return [str(path) for path in [*bundle_paths, *table_paths]], [
                {"run_url": getattr(run, "url", None)}
            ]

    if stage == "evaluate_discriminator_metrics":
        require_files([cfg.real_nc_file], stage)
        checkpoint_root = Path(str(baseline_get(cfg, "discriminator", {})["checkpoint_dir"]))
        if not checkpoint_root.is_dir():
            raise FileNotFoundError(
                f"Stage {stage} requires local checkpoints under {checkpoint_root}"
            )
        with tracker.run("evaluation/discriminator-metrics", "discriminator-metric-evaluation", cfg,
                         tags=["evaluation", "discriminator-metrics"]) as run:
            if tracker.enabled:
                for artifact in tracker.logged_artifacts:
                    if getattr(artifact, "type", None) == "model":
                        run.use_artifact(artifact)
            paths = evaluate_discriminator_metrics(cfg)
            csv_path = output_root / "data" / "discriminator_reverse_kl.csv"
            tracker.log_csv_table(run, "metrics/discriminator_reverse_kl", csv_path)
            if csv_path.is_file():
                run.summary["metrics/discriminator_rows"] = csv_row_count(csv_path)
            if upload_data:
                tracker.log_artifact(
                    run, "discriminator-metric-evaluation", "evaluation", [*paths, resolved_path],
                    metadata={"pipeline_id": tracker.group},
                )
            return [str(path) for path in paths], [{"run_url": getattr(run, "url", None)}]

    if stage == "plot":
        require_files(plot_input_paths(cfg, output_root), stage)
        before = {
            path: path.stat().st_mtime_ns
            for path in (output_root / "plots").rglob("*.png")
        } if (output_root / "plots").exists() else {}
        with tracker.run("plotting", "plotting", cfg, tags=["plotting"]) as run:
            plot_saved_standard_metric_baselines(cfg)
            interpretability_paths, cases_path = plot_target_discriminator_interpretability(cfg)
            paths = changed_plot_paths(output_root, before)
            generated_paths = {Path(path) for path in interpretability_paths}
            standard_paths = [path for path in paths if path not in generated_paths]
            tracker.log_images(run, standard_paths, output_root / "plots")
            target_root = Path(str(cfg.target_discriminator.output_dir))
            tracker.log_images(run, interpretability_paths, target_root / "plots")
            if cases_path is not None:
                tracker.log_csv_table(run, "interpretability/cases", cases_path)
            rendered_paths = list(dict.fromkeys([*paths, *interpretability_paths]))
            all_paths = plot_bundle_members(rendered_paths)
            run.summary["plots/count"] = len(rendered_paths)
            run.summary["interpretability/galleries"] = len(interpretability_paths)
            if cases_path is not None and Path(cases_path).is_file():
                run.summary["interpretability/case_rows"] = csv_row_count(cases_path)
            if bool(cfg.pipeline.wandb.get("upload_plots", True)):
                tracker.log_artifact(
                    run, "baseline-plots", "plots", [*all_paths, *([cases_path] if cases_path is not None else []), resolved_path],
                    metadata={"pipeline_id": tracker.group},
                )
            return [str(path) for path in all_paths], [{"run_url": getattr(run, "url", None)}]

    raise AssertionError(f"Unhandled pipeline stage: {stage}")


def execute_pipeline(cfg):
    configure_plot_bundle_saving_from_cfg(cfg)
    pipeline_id = safe_pipeline_id = str(cfg.pipeline.id or generated_pipeline_id())
    # Keep local directory names portable and exactly aligned with W&B names.
    pipeline_id = safe_name(pipeline_id)
    stages = selected_stages(cfg)
    histogram_enabled = bool((cfg.get("histogram_matching", {}) or {}).get("enabled", False))
    if histogram_enabled and "fit_histogram_matching" not in stages:
        input_maps = cfg.pipeline.get("input_histogram_matching_dir")
        if input_maps is None:
            raise ValueError(
                "Histogram matching is enabled without fit_histogram_matching; "
                "set pipeline.input_histogram_matching_dir to an existing fitted artifact."
            )
        require_files([Path(str(input_maps)) / "maps.npz",
                       Path(str(input_maps)) / "manifest.json"], "histogram_matching")
    runs_parent = pipeline_runs_dir(cfg)
    run_dir = runs_parent / pipeline_id
    resume = bool(cfg.pipeline.get("resume", False))
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"Pipeline run directory already exists: {run_dir}. "
            "Choose a new pipeline.id or set pipeline.resume=true to reuse it intentionally."
        )
    run_dir.mkdir(parents=True, exist_ok=resume)
    configure_run_output_dirs(cfg, pipeline_id, run_dir, runs_parent)
    resolved_path = run_dir / "resolved_config.yaml"
    manifest_path = run_dir / "manifest.json"
    OmegaConf.save(cfg, resolved_path, resolve=True)
    tracker = PipelineTracker(cfg, pipeline_id)
    variables = variables_from_config(cfg)
    output_root = baseline_output_dir(cfg, variables)
    records = []
    first_error = None

    for stage in stages:
        started = time.monotonic()
        record = {"stage": stage, "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            outputs, details = run_stage(stage, cfg, tracker, output_root, resolved_path)
            record.update(status="completed", outputs=outputs, details=details)
        except BaseException as error:
            record.update(
                status="failed", error=f"{type(error).__name__}: {error}",
                traceback="".join(traceback.format_exception(error)), outputs=[], details=[],
            )
            first_error = first_error or error
        record["duration_seconds"] = time.monotonic() - started
        records.append(record)
        manifest_path.write_text(json.dumps({"pipeline_id": pipeline_id, "stages": records}, indent=2))
        if first_error is not None and bool(cfg.pipeline.get("fail_fast", True)):
            break

    manifest = {
        "pipeline_id": pipeline_id,
        "requested_pipeline_id": safe_pipeline_id,
        "run_dir": str(run_dir),
        "status": "failed" if first_error is not None else "completed",
        "selected_stages": stages,
        "stages": records,
        "resolved_config": str(resolved_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    with tracker.run("pipeline-summary", "pipeline-summary", cfg, tags=["summary"]) as run:
        summary_records = [
            {
                "stage": record["stage"], "status": record["status"],
                "duration_seconds": record["duration_seconds"],
                "error": record.get("error", ""),
            }
            for record in records
        ]
        tracker.log_records_table(run, "pipeline/stages", summary_records)
        run.summary["pipeline/status"] = manifest["status"]
        run.summary["pipeline/stages_completed"] = sum(r["status"] == "completed" for r in records)
        run.summary["pipeline/run_dir"] = str(run_dir)
        tracker.log_artifact(run, "pipeline-manifest", "pipeline", [manifest_path, resolved_path])
    if first_error is not None:
        raise first_error
    return manifest


@hydra.main(version_base=None, config_path="../conf", config_name="baseline_pipeline")
def main(cfg: DictConfig):
    execute_pipeline(cfg)


if __name__ == "__main__":
    main()
