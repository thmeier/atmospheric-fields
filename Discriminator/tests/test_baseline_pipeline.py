import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from Discriminator.scripts.baseline_pipeline_tracking import (
    PipelineTracker,
    parsed_csv_value,
    safe_name,
)
from Discriminator.scripts.run_baseline_pipeline import execute_pipeline, selected_stages


def pipeline_config(root, stages):
    return OmegaConf.create({
        "baseline": {
            "variables": ["2m_temperature"],
            "output_dir": str(root / "baseline"),
        },
        "target_discriminator": {
            "output_dir": str(root / "baseline" / "2m_temperature"),
            "checkpoint_dir": None,
        },
            "pipeline": {
                "id": "fixture-run",
                "stages": stages,
                "fail_fast": True,
                "runs_dir": str(root / "runs"),
                "resume": False,
            "wandb": {
                "enabled": False,
                "project": "fixture",
                "entity": None,
                "mode": "disabled",
                "tags": [],
            },
        },
    })


class BaselinePipelineTests(unittest.TestCase):
    def test_stages_are_validated_and_run_in_canonical_order(self):
        cfg = pipeline_config(Path("/tmp"), ["plot", "train_discriminators"])
        self.assertEqual(selected_stages(cfg), ["train_discriminators", "plot"])
        cfg.pipeline.stages = ["plot", "plot"]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            selected_stages(cfg)
        cfg.pipeline.stages = ["upload"]
        with self.assertRaisesRegex(ValueError, "unknown"):
            selected_stages(cfg)

    def test_disabled_pipeline_writes_reproducibility_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = pipeline_config(root, ["plot", "evaluate_standard_metrics"])
            calls = []

            def fake_stage(stage, _cfg, _tracker, _output_root, resolved_path):
                calls.append(stage)
                self.assertTrue(Path(resolved_path).is_file())
                return [f"{stage}.artifact"], [{"run_url": None}]

            with patch("Discriminator.scripts.run_baseline_pipeline.run_stage", fake_stage):
                manifest = execute_pipeline(cfg)

            self.assertEqual(calls, ["evaluate_standard_metrics", "plot"])
            self.assertEqual(manifest["status"], "completed")
            manifest_path = root / "runs" / "fixture-run" / "manifest.json"
            saved = json.loads(manifest_path.read_text())
            self.assertEqual(saved["selected_stages"], calls)
            self.assertTrue(Path(saved["resolved_config"]).is_file())
            self.assertEqual(Path(saved["run_dir"]), root / "runs" / "fixture-run")
            self.assertEqual(
                Path(str(cfg.baseline.output_dir)), root / "runs" / "fixture-run",
            )
            resolved = OmegaConf.load(manifest_path.parent / "resolved_config.yaml")
            self.assertEqual(Path(str(resolved.pipeline.runs_dir)), root / "runs")

    def test_existing_pipeline_id_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = pipeline_config(root, [])
            (root / "runs" / "fixture-run").mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "pipeline.resume=true"):
                execute_pipeline(cfg)

    def test_explicit_checkpoint_input_is_preserved_while_outputs_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "previous-run" / "2m_temperature" / "models" / "target_discriminators"
            cfg = pipeline_config(root, [])
            cfg.pipeline.input_checkpoint_dir = str(source)
            execute_pipeline(cfg)
            self.assertEqual(Path(str(cfg.target_discriminator.output_dir)), root / "runs" / "fixture-run" / "2m_temperature")
            self.assertEqual(Path(str(cfg.target_discriminator.checkpoint_dir)), source)
            cfg.pipeline.resume = True
            with patch("Discriminator.scripts.run_baseline_pipeline.run_stage"):
                execute_pipeline(cfg)

    def test_disabled_tracker_is_a_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = pipeline_config(Path(directory), [])
            tracker = PipelineTracker(cfg, "fixture-run")
            with tracker.run("stage", "test", cfg) as run:
                run.log({"loss": 1.0})
                run.summary["example"] = 1
            self.assertFalse(tracker.enabled)
            self.assertEqual(run.summary["example"], 1)

    def test_wandb_names_and_csv_values_are_stable(self):
        self.assertEqual(safe_name("train/GraphCast +6h"), "train-GraphCast-6h")
        self.assertEqual(parsed_csv_value("7"), 7)
        self.assertEqual(parsed_csv_value("0.25"), 0.25)
        self.assertIs(parsed_csv_value("false"), False)
        self.assertEqual(parsed_csv_value("GraphCast"), "GraphCast")

    def test_tracker_uses_pipeline_directory_for_wandb_local_files(self):
        class FakeRun:
            def __init__(self):
                self.summary = {}

            def finish(self, **_kwargs):
                pass

        class FakeWandb:
            def __init__(self):
                self.init_kwargs = None

            def init(self, **kwargs):
                self.init_kwargs = kwargs
                return FakeRun()

        with tempfile.TemporaryDirectory() as directory:
            cfg = pipeline_config(Path(directory), [])
            cfg.pipeline.wandb.enabled = True
            cfg.pipeline.wandb.mode = "disabled"
            cfg.pipeline.run_dir = str(Path(directory) / "runs" / "fixture-run")
            tracker = PipelineTracker(cfg, "fixture-run")
            fake_wandb = FakeWandb()
            tracker._wandb = fake_wandb
            with tracker.run("plotting", "plots", cfg):
                pass
            self.assertEqual(fake_wandb.init_kwargs["name"], "fixture-run/plotting")
            self.assertEqual(fake_wandb.init_kwargs["dir"], str(Path(cfg.pipeline.run_dir) / "wandb"))


if __name__ == "__main__":
    unittest.main()
