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
        "pipeline": {
            "id": "fixture-run",
            "stages": stages,
            "fail_fast": True,
            "manifest_dir": str(root / "manifests"),
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
            manifest_path = root / "manifests" / "fixture-run" / "manifest.json"
            saved = json.loads(manifest_path.read_text())
            self.assertEqual(saved["selected_stages"], calls)
            self.assertTrue(Path(saved["resolved_config"]).is_file())

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


if __name__ == "__main__":
    unittest.main()
