#!/usr/bin/env python3
"""Download NOAA GEFS forecasts from Dynamical Catalog.

This is the GEFS counterpart to ``download_gfs_dynamical_netcdf.py``.  It uses
the same project defaults:

- surface fields: 2m temperature, 10m U/V wind, mean sea-level pressure
- lead times: 6, 12, 24, 48, 96, and 192 hours
- init-time end: 2023-12-31T23:59:59

Use a ``.zarr`` output path for an intermediate suitable for WeatherBench2
regridding, or a ``.nc``/``.netcdf`` path for a final NetCDF file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from download_gfs_dynamical_netcdf import (
    DEFAULT_LEAD_HOURS,
    DEFAULT_TIME_END,
    download_dynamical_forecast,
    _parse_ensemble_members,
)


DATASET_NAME = "noaa-gefs-forecast-35-day"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output .zarr/.nc file path")
    parser.add_argument(
        "--dataset",
        default=DATASET_NAME,
        help=f"Dynamical Catalog dataset name (default: {DATASET_NAME})",
    )
    parser.add_argument(
        "-s",
        "--time-start",
        default=None,
        help="Optional init-time start, e.g. 2020-01-01T00. Default: dataset start.",
    )
    parser.add_argument(
        "-e",
        "--time-end",
        default=DEFAULT_TIME_END,
        help=f"Optional init-time end. Default: {DEFAULT_TIME_END}.",
    )
    parser.add_argument(
        "--lead-hours",
        nargs="+",
        type=int,
        default=list(DEFAULT_LEAD_HOURS),
        help="Forecast lead times in hours to keep.",
    )
    parser.add_argument(
        "--ensemble-members",
        type=_parse_ensemble_members,
        default=None,
        help="Comma-separated ensemble members to keep, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--keep-source-names",
        action="store_true",
        help="Do not rename Dynamical source variables to repo-compatible names.",
    )
    args = parser.parse_args()

    download_dynamical_forecast(
        output_path=args.output,
        dataset_name=args.dataset,
        time_start=args.time_start,
        time_end=args.time_end,
        lead_hours=args.lead_hours,
        ensemble_members=args.ensemble_members,
        keep_source_names=args.keep_source_names,
    )


if __name__ == "__main__":
    main()
