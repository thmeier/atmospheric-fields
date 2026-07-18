#!/usr/bin/env python3
"""Download ECMWF AIFS Single forecasts from Dynamical Catalog.

Defaults match the project's surface forecast setup:

- dataset: ecmwf-aifs-single-forecast
- init times: 2025-01-01T00 through 2025-12-31T18, i.e. [2025, 2026)
- surface fields: 2m temperature, 10m U/V wind, mean sea-level pressure
- lead times: 6, 12, 24, 48, 96, and 192 hours

AIFS exposes 2m temperature in degree Celsius.  By default this script converts
it to Kelvin after renaming to the repo-compatible ``2m_temperature`` name.
Pass ``--keep-source-units`` to leave source units unchanged.

Use a ``.zarr`` output path for a native-resolution intermediate suitable for
WeatherBench2 regridding, or a ``.nc``/``.netcdf`` path for a direct NetCDF file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from download_gfs_dynamical_netcdf import (
    DEFAULT_LEAD_HOURS,
    _import_dynamical_catalog,
    _convert_2m_temperature_to_kelvin,
    _drop_regrid_incompatible_vars,
    _resolve_variables,
    _select_init_time,
    _select_lead_times,
    _write_dataset,
)


DATASET_NAME = "ecmwf-aifs-single-forecast"
DEFAULT_TIME_START = "2025-01-01T00"
DEFAULT_TIME_END = "2025-12-31T18"
def download_aifs_dynamical_forecast(
    output_path: Path,
    dataset_name: str,
    time_start: str | None,
    time_end: str | None,
    lead_hours: list[int],
    keep_source_names: bool,
    keep_source_units: bool,
):
    dynamical_catalog = _import_dynamical_catalog()

    print(f"Opening Dynamical Catalog dataset {dataset_name!r}...")
    ds = dynamical_catalog.open(dataset_name)
    print(f"Available variables: {list(ds.data_vars)}")
    print(f"Dimensions: {dict(ds.dims)}")

    variables, rename = _resolve_variables(ds, keep_source_names=keep_source_names)
    print(f"Selecting variables: {variables}")
    ds = ds[variables]
    if rename:
        print(f"Renaming source variables for repo compatibility: {rename}")
        ds = ds.rename(rename)

    if keep_source_units:
        print("Keeping source units unchanged.")
    else:
        ds = _convert_2m_temperature_to_kelvin(ds)

    ds = _drop_regrid_incompatible_vars(ds)
    ds = _select_lead_times(ds, lead_hours)
    ds = _select_init_time(ds, time_start, time_end)

    _write_dataset(ds, output_path)
    print("Done!")


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
        default=DEFAULT_TIME_START,
        help=f"Optional init-time start. Default: {DEFAULT_TIME_START}.",
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
        "--keep-source-names",
        action="store_true",
        help="Do not rename Dynamical source variables to repo-compatible names.",
    )
    parser.add_argument(
        "--keep-source-units",
        action="store_true",
        help="Do not convert AIFS 2m temperature from Celsius to Kelvin.",
    )
    args = parser.parse_args()

    download_aifs_dynamical_forecast(
        output_path=args.output,
        dataset_name=args.dataset,
        time_start=args.time_start,
        time_end=args.time_end,
        lead_hours=args.lead_hours,
        keep_source_names=args.keep_source_names,
        keep_source_units=args.keep_source_units,
    )


if __name__ == "__main__":
    main()
