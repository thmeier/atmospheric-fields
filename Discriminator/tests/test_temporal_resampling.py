import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from Discriminator.scripts.analyze_temporal_resamples import (
    compare_null_to_coordinate, reconstruct_scores_from_terms,
)
from Discriminator.scripts.temporal_resampling import (
    TemporalSchedule, aggregate_draws, fixed_schedules, learned_schedules, schedule_mask, write_csv_gz,
)


def config():
    return OmegaConf.create({
        "temporal_resampling": {
            "enabled": True, "learned_replicates": 5, "fixed_replicates": 50,
            "learned_test_windows": [[5, 11], [9, 15], [13, 19], [17, 23], [20, 26]],
            "buffer_days": 4, "random_test_start_range": [5, 20], "seed": 7,
            "active_schedule": None,
        }
    })


class TemporalResamplingTests(unittest.TestCase):
    def test_learned_windows_and_buffer_are_disjoint(self):
        schedule = learned_schedules(config())[0]
        values = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-02-01"))
        ranges = [(np.datetime64("2020-01-01"), np.datetime64("2020-01-31"))]
        test = schedule_mask(values, schedule, "test", ranges)
        train = schedule_mask(values, schedule, "train", ranges)
        self.assertEqual(values[test][0], np.datetime64("2020-01-05"))
        self.assertEqual(values[test][-1], np.datetime64("2020-01-11"))
        self.assertFalse(np.any(test & train))
        self.assertFalse(np.any(train[:15]))
        self.assertTrue(np.all(train[15:]))

    def test_fixed_schedules_are_deterministic_and_keep_seven_day_tests(self):
        ranges = [("2020-01-01", "2020-12-31")]
        first = fixed_schedules(config(), ranges)
        second = fixed_schedules(config(), ranges)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertEqual(first[4].test_days, (20, 26))
        for schedule in first[5:]:
            for month, _ in schedule.monthly_test_starts:
                lower, upper = schedule.test_window(month)
                self.assertEqual(upper - lower + 1, 7)
                self.assertGreaterEqual(lower, 5)
                self.assertLessEqual(upper, 26)

    def test_aggregates_preserve_draw_ids_and_requested_bounds(self):
        rows = [
            {"target": "GraphCast", "x": "12", "resample_id": f"fold_{i}", "score": value}
            for i, value in enumerate([1, 2, 3, 4, 10])
        ]
        result = aggregate_draws(rows, ["score"], ["target", "x"], "minmax")[0]
        self.assertEqual(result["score"], 4.0)
        self.assertEqual(result["score_lower"], 1.0)
        self.assertEqual(result["score_upper"], 10.0)
        self.assertEqual(result["n_resamples"], 5)
        self.assertEqual(result["draw_ids"], "fold_0,fold_1,fold_2,fold_3,fold_4")

    def test_query_counts_null_draws_above_lead_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draws.csv"
            fields = ["architecture", "kind", "target", "x", "is_era5_test_null",
                      "score", "resample_id"]
            records = []
            for index, score in enumerate([1, 2, 3, 4, 5]):
                records.append(dict(zip(fields, ["squeezenet", "forecast", "GraphCast", 0,
                                                 True, score, f"fold_{index}"])))
            for index, score in enumerate([2, 2, 2, 2, 2]):
                records.append(dict(zip(fields, ["squeezenet", "forecast", "GraphCast", 12,
                                                 False, score, f"fold_{index}"])))
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(records)
            result = compare_null_to_coordinate(path, "squeezenet", "GraphCast", 12)
            self.assertEqual(result["candidate_mean"], 2.0)
            self.assertEqual(result["null_exceeding_count"], 3)
            self.assertEqual(result["null_count"], 5)

    def test_raw_terms_exactly_reconstruct_reverse_kl_draw(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terms.csv.gz"
            common = {"resample_id": "learned_00", "architecture": "squeezenet",
                      "kind": "forecast", "target": "GraphCast"}
            terms = [
                {**common, "role": "ep_train", "source": "ERA5 train", "x": "",
                 "transformed_term": value} for value in (-2.0, -4.0)
            ] + [
                {**common, "role": "candidate", "source": "GraphCast", "x": 12,
                 "transformed_term": value} for value in (-5.0, -7.0)
            ]
            write_csv_gz(path, terms)
            reconstructed = reconstruct_scores_from_terms(path)
            key = ("learned_00", "squeezenet", "forecast", "GraphCast", "GraphCast", 12.0)
            self.assertEqual(reconstructed[key], 3.0)


if __name__ == "__main__":
    unittest.main()


class CountSpreadTests(unittest.TestCase):
    """Sample counts differ per fold wherever no evaluation cap binds."""

    @staticmethod
    def _rows(counts):
        return [
            {"label": "GraphCast", "lead_hour": "192", "resample_id": f"fixed_{i:03d}",
             "n_samples": str(n), "pairwise_n_samples": "256"}
            for i, n in enumerate(counts)
        ]

    def test_equal_counts_report_no_spread(self):
        from Discriminator.scripts.temporal_resampling_pipeline import _validate_counts
        spreads = _validate_counts(self._rows([1000, 1000, 1000]),
                                   ["label", "lead_hour"], ("n_samples", "pairwise_n_samples"))
        self.assertEqual(spreads, {})

    def test_varying_forecast_counts_are_recorded_not_fatal(self):
        # A +192 h forecast pair survives only if its valid time lands inside the
        # same seven-day window, so the count depends on where the window fell.
        from Discriminator.scripts.temporal_resampling_pipeline import _validate_counts
        spreads = _validate_counts(self._rows([160, 164, 166, 168]),
                                   ["label", "lead_hour"], ("n_samples", "pairwise_n_samples"))
        self.assertEqual(spreads[(("GraphCast", "192"), "n_samples")], (160, 168))
        # The capped pairwise count is identical, so it must not be flagged.
        self.assertNotIn((("GraphCast", "192"), "pairwise_n_samples"), spreads)


class CanonicalPlotCopyTests(unittest.TestCase):
    """Training figures must be found under either profile and variable tag."""

    def test_finds_plots_under_paper_profile_and_a_different_variable_tag(self):
        import shutil as _shutil
        from unittest import mock
        from Discriminator.scripts import temporal_resampling_pipeline as trp
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            # Training writes under its own tag, nested under the paper profile.
            deep = child / "t2m__u10__v10__msl" / "plots" / "paper" / "target_logit_distributions"
            (deep / "squeezenet" / "forecast" / "GraphCast").mkdir(parents=True)
            (deep / "squeezenet" / "forecast" / "GraphCast" / "all_lead_times.png").write_text("x")
            # A wandb media mirror must never be treated as a source.
            noise = child / "wandb" / "plots" / "target_logit_distributions"
            noise.mkdir(parents=True)
            parent = root / "parent"
            with mock.patch.object(trp, "_child_run_dir", return_value=child), \
                 mock.patch.object(trp, "_parent_output_root", return_value=parent):
                copied = trp.copy_canonical_training_plots(
                    None, TemporalSchedule("learned_04", "learned", 4, (20, 26)))
            self.assertTrue(copied, "expected the paper-profile plots to be found")
            landed = parent / "plots" / "paper" / "target_logit_distributions" / \
                "squeezenet" / "forecast" / "GraphCast" / "all_lead_times.png"
            self.assertTrue(landed.is_file(), f"missing {landed}")


class UnconvergedCriticTests(unittest.TestCase):
    """A critic at chance accuracy must leave the sample, not be averaged in."""

    @staticmethod
    def _fold(root, resample_id, accuracies):
        data = root / resample_id / "some__variable__tag" / "data"
        data.mkdir(parents=True)
        with open(data / "target_train_test_metrics.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["architecture", "kind", "target",
                                                        "train_accuracy", "test_accuracy"])
            writer.writeheader()
            for target, acc in accuracies.items():
                writer.writerow({"architecture": "squeezenet", "kind": "forecast",
                                 "target": target, "train_accuracy": "0.88",
                                 "test_accuracy": str(acc)})

    def _setup(self, directory, threshold=0.6):
        from unittest import mock
        from Discriminator.scripts import temporal_resampling_pipeline as trp
        root = Path(directory)
        # learned_01's GraphCast critic never converged; everything else did.
        self._fold(root, "learned_00", {"GraphCast": 0.86, "FuXi": 0.90})
        self._fold(root, "learned_01", {"GraphCast": 0.52, "FuXi": 0.89})
        schedules = [TemporalSchedule("learned_00", "learned", 0, (5, 11)),
                     TemporalSchedule("learned_01", "learned", 1, (9, 15))]
        rows = [
            {"resample_id": "learned_00", "architecture": "squeezenet", "kind": "forecast",
             "target": "GraphCast", "score": "12.6"},
            {"resample_id": "learned_01", "architecture": "squeezenet", "kind": "forecast",
             "target": "GraphCast", "score": "-230.6"},
            {"resample_id": "learned_01", "architecture": "squeezenet", "kind": "forecast",
             "target": "FuXi", "score": "18.8"},
        ]
        cfg = OmegaConf.create({"temporal_resampling": {"min_critic_test_accuracy": threshold}})
        patch = mock.patch.object(trp, "_child_run_dir",
                                  side_effect=lambda c, s: root / s.resample_id)
        return trp, cfg, rows, schedules, patch

    def test_chance_level_critic_is_dropped_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            trp, cfg, rows, schedules, patch = self._setup(directory)
            with patch:
                kept, dropped = trp.drop_unconverged_critics(cfg, rows, schedules)
            self.assertEqual(len(kept), 2)
            self.assertEqual(len(dropped), 1)
            self.assertEqual(dropped[0]["target"], "GraphCast")
            self.assertEqual(dropped[0]["resample_id"], "learned_01")
            # Only that one critic goes; the same fold's converged critic stays.
            self.assertIn(("learned_01", "FuXi"),
                          {(r["resample_id"], r["target"]) for r in kept})

    def test_threshold_of_none_keeps_every_draw(self):
        with tempfile.TemporaryDirectory() as directory:
            trp, cfg, rows, schedules, patch = self._setup(directory, threshold=None)
            with patch:
                kept, dropped = trp.drop_unconverged_critics(cfg, rows, schedules)
            self.assertEqual(len(kept), 3)
            self.assertEqual(dropped, [])

    def test_exclusion_never_inspects_the_score(self):
        # A wildly negative score from a CONVERGED critic must be retained;
        # the rule is about convergence, not about disliking an outcome.
        with tempfile.TemporaryDirectory() as directory:
            trp, cfg, rows, schedules, patch = self._setup(directory)
            rows[0]["score"] = "-999.0"          # learned_00 GraphCast, accuracy 0.86
            with patch:
                kept, _ = trp.drop_unconverged_critics(cfg, rows, schedules)
            self.assertIn("-999.0", {r["score"] for r in kept})
