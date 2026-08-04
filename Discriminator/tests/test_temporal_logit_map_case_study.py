import unittest

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf

from Discriminator.scripts.monthly_split import forecast_pairs, select_era5_split
from Discriminator.scripts.plot_temporal_logit_map_case_study import matching_era5_index, timestamp


class TemporalLogitMapCaseStudyTest(unittest.TestCase):
    def test_timestamp_accepts_scalar_and_singleton_xarray_values(self):
        expected = pd.Timestamp("2018-08-31 00:00:00")
        self.assertEqual(timestamp(np.datetime64("2018-08-31")), expected)
        self.assertEqual(timestamp(np.array([np.datetime64("2018-08-31")])), expected)

    def test_timestamp_rejects_ambiguous_multiple_values(self):
        with self.assertRaisesRegex(ValueError, "Expected one timestamp"):
            timestamp(np.array(["2018-08-31", "2018-09-01"], dtype="datetime64[D]"))

    def test_matching_era5_index_accepts_object_datetime_array(self):
        era5_times = np.array(
            [
                pd.Timestamp("2018-08-31 00:00").to_pydatetime(),
                pd.Timestamp("2018-08-31 06:00").to_pydatetime(),
            ],
            dtype=object,
        )
        with self.assertRaisesRegex(ValueError, "exact ERA5 sample"):
            matching_era5_index(era5_times, pd.Timestamp("2018-08-31 05:00"))
        index, alignment_hours = matching_era5_index(
            era5_times, pd.Timestamp("2018-08-31 06:00")
        )
        self.assertEqual(index, 1)
        self.assertEqual(alignment_hours, 0.0)


class MonthlyValidTimeSplitTest(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create({"monthly_split": {
            "strategy": "monthly_valid_time",
            "train_days": [1, 15],
            "test_days": [20, 26],
            "null_comparison_days": [16, 31],
            "corruption_time_range": ["2020-01-01", "2020-12-31"],
            "model_valid_time_ranges": [["2020-01-01", "2020-12-31"]],
            "require_exact_era5_match": True,
        }})
        times = np.arange(
            np.datetime64("2020-01-01T00"), np.datetime64("2021-01-01T00"),
            np.timedelta64(6, "h"),
        )
        self.era5 = xr.Dataset({"x": ("time", np.arange(len(times)))}, coords={"time": times})

    def test_calendar_memberships_are_disjoint_and_match_config(self):
        train = set(np.asarray(select_era5_split(self.era5, self.cfg, "train").time.dt.day))
        test = set(np.asarray(select_era5_split(self.era5, self.cfg, "test").time.dt.day))
        null = set(np.asarray(select_era5_split(self.era5, self.cfg, "null").time.dt.day))
        self.assertEqual(train, set(range(1, 16)))
        self.assertEqual(test, set(range(20, 27)))
        self.assertEqual(null, set(range(16, 32)))
        self.assertFalse(train & test)

    def test_forecasts_are_split_by_valid_time_not_initialization_time(self):
        forecast = xr.Dataset(
            {"x": (("time", "prediction_timedelta"), np.zeros((3, 3)))},
            coords={
                "time": np.array(["2020-01-19T12", "2020-01-20T00", "2020-01-26T00"], dtype="datetime64[h]"),
                "prediction_timedelta": np.array([6, 24, 36], dtype="timedelta64[h]"),
            },
        )
        pairs = forecast_pairs(forecast, self.era5, self.cfg, "test")
        actual = {(pair.forecast_index, pair.lead_hour) for pair in pairs}
        self.assertIn((0, 24), actual)
        self.assertNotIn((0, 6), actual)
        self.assertIn((2, 6), actual)
        self.assertNotIn((2, 24), actual)
        self.assertTrue(all(pair.valid_time == self.era5.time.values[pair.era5_index] for pair in pairs))


if __name__ == "__main__":
    unittest.main()
