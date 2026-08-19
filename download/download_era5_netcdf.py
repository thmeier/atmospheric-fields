#!/usr/bin/env python3
"""Select a WeatherBench2 Zarr time window and write it as NetCDF.

This helper is intentionally not a regridder. Use an already suitable
WeatherBench2 source, for example a 240x121 1.5-degree store when downstream
code expects the WeatherBench2 paper/evaluation grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from zarr_to_netcdf import _sanitize_dataset_attrs


DEFAULT_LEAD_HOURS = (6, 12, 24, 48, 96, 192)
LEAD_TIME_CANDIDATES = (
    "prediction_timedelta",
    "lead_time",
    "lead_times",
    "step",
    "forecast_hour",
    "time_delta",
)


def _open_zarr(source: str):
    import xarray as xr

    storage_options = {"token": "anon"} if source.startswith("gs://") else None
    return xr.open_zarr(source, chunks={}, storage_options=storage_options)


def _available_hours(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[h]").astype(int)
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float).round().astype(int)

    hours = []
    for value in values:
        text = str(value)
        if text.endswith("h"):
            hours.append(int(text[:-1]))
        else:
            hours.append(int(text))
    return np.asarray(hours, dtype=int)


def _find_lead_coord(ds) -> str | None:
    for name in LEAD_TIME_CANDIDATES:
        if name in ds.dims or name in ds.coords:
            return name
    for name in list(ds.dims) + list(ds.coords):
        lower = name.lower()
        if any(candidate in lower for candidate in LEAD_TIME_CANDIDATES):
            return name
    return None


def _select_variables(ds, variables: list[str] | None):
    if not variables:
        return ds

    missing = [name for name in variables if name not in ds]
    if missing:
        raise SystemExit(
            f"Variables not found: {missing}. Available variables: {list(ds.data_vars)}"
        )
    return ds[variables]


def _select_levels(ds, levels: list[float] | None):
    if not levels:
        return ds
    if "level" not in ds.dims and "level" not in ds.coords:
        raise SystemExit("--level was supplied, but the dataset has no 'level' coordinate.")

    selected = int(levels[0]) if len(levels) == 1 and levels[0].is_integer() else levels
    print(f"Filtering level to {selected}")
    return ds.sel(level=selected)


def _select_lead_times(ds, lead_hours: list[int] | None):
    lead_coord = _find_lead_coord(ds)
    if lead_coord is None:
        print("No lead time dimension detected. Proceeding without lead-time filtering.")
        return ds
    if not lead_hours:
        print(f"Detected {lead_coord}; keeping all lead times.")
        return ds

    values = ds[lead_coord].values
    hours = _available_hours(values)
    selected_values = []
    missing = []
    for lead_hour in lead_hours:
        matches = np.where(hours == lead_hour)[0]
        if len(matches) == 0:
            missing.append(lead_hour)
        else:
            selected_values.append(values[int(matches[0])])

    if missing:
        print(f"Warning: requested lead times not found: {missing}h")
    if not selected_values:
        raise SystemExit(
            f"None of the requested lead times {lead_hours}h are available. "
            f"Available lead times: {hours.tolist()}h"
        )

    selected_hours = _available_hours(np.asarray(selected_values)).tolist()
    print(f"Filtering {lead_coord} to lead times: {selected_hours}h")
    return ds.sel({lead_coord: selected_values})


def _parse_lead_hours(value: str) -> list[int] | None:
    if value.lower() in {"all", "none"}:
        return None
    return [int(part) for part in value.replace(",", " ").split() if part]


def download_era5_netcdf(
    output_path: Path,
    source: str,
    variables: list[str] | None,
    time_start: str,
    time_end: str,
    levels: list[float] | None,
    lead_hours: list[int] | None,
    output_format: str = "netcdf",
):
    print(f"Opening {source}...")
    ds = _open_zarr(source)
    print(f"Available variables: {list(ds.data_vars)}")
    print(f"Dimensions: {dict(ds.dims)}")

    ds = _select_variables(ds, variables)
    ds = ds.sel(time=slice(time_start, time_end))
    ds = _select_levels(ds, levels)
    ds = _select_lead_times(ds, lead_hours)

    if "time" in ds.sizes and ds.sizes["time"] == 0:
        raise SystemExit(f"No data in time range {time_start} through {time_end}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = _sanitize_dataset_attrs(ds)
    print(f"Saving to {output_path}")
    if output_format == "zarr":
        # Keep the WeatherBench2 store layout so downstream code that expects the
        # upstream zarr (e.g. U-Cast's data module) can read it with only a path change.
        #
        # Two pieces of inherited encoding have to be dealt with first:
        #  1. zarr_format=2 is required, not cosmetic. The upstream store is v2 and its
        #     variables carry numcodecs Blosc compressors in .encoding. zarr-python 3.x
        #     defaults to writing v3, which rejects a v2 codec with
        #     "Expected a BytesBytesCodec. Got <class 'numcodecs.blosc.Blosc'>".
        #  2. encoding['chunks'] is inherited as (8, ...) along time. A month-long slice
        #     rarely starts on one of those 8-step boundaries, so the dask chunks straddle
        #     zarr chunks and to_zarr refuses ("would overlap multiple Dask chunks").
        #     Rechunk time uniformly and drop the stale chunk encoding so the written
        #     chunks are derived from the dask layout instead.
        if "time" in ds.dims:
            ds = ds.chunk({"time": 8})
        for name in ds.variables:
            ds[name].encoding.pop("chunks", None)
            ds[name].encoding.pop("preferred_chunks", None)
        ds.to_zarr(output_path, mode="w", consolidated=True, zarr_format=2)
    else:
        ds.to_netcdf(output_path, format="NETCDF4")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="WeatherBench2 Zarr path or URL")
    parser.add_argument("output", type=Path, help="Output .nc/.netcdf file or .zarr store")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("netcdf", "zarr"),
        default="netcdf",
        help="Output format. Default: netcdf.",
    )
    parser.add_argument("-v", "--variables", nargs="+")
    parser.add_argument("-s", "--time-start", required=True)
    parser.add_argument("-e", "--time-end", required=True)
    parser.add_argument(
        "--level",
        nargs="+",
        type=float,
        default=None,
        help="Optional pressure level(s) to select, e.g. --level 850.",
    )
    parser.add_argument(
        "--lead-hours",
        type=_parse_lead_hours,
        default=list(DEFAULT_LEAD_HOURS),
        help=(
            "Forecast lead hours to keep, as a quoted space/comma-separated list. "
            "Use 'all' to keep all lead times. Default: 6 12 24 48 96 192."
        ),
    )
    args = parser.parse_args()

    download_era5_netcdf(
        output_path=args.output,
        source=args.source,
        variables=args.variables,
        time_start=args.time_start,
        time_end=args.time_end,
        levels=args.level,
        lead_hours=args.lead_hours,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
