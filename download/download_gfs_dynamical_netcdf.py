#!/usr/bin/env python3
"""Download NOAA GFS forecasts from Dynamical Catalog.

Defaults match the project forecast setup:

- surface fields: 2m temperature, 10m U/V wind, mean sea-level pressure
- lead times: 6, 12, 24, 48, 96, and 192 hours

The Dynamical Catalog GFS dataset uses names such as ``temperature_2m`` and
``wind_u_10m`` for surface fields, while the rest of this repository uses
WeatherBench-style names such as ``2m_temperature`` and
``10m_u_component_of_wind``.  By default this script writes repo-compatible
names where possible; pass ``--keep-source-names`` to preserve the source names.

Use a ``.zarr`` output path for an intermediate suitable for WeatherBench2
regridding, or a ``.nc``/``.netcdf`` path for a final NetCDF file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


DATASET_NAME = "noaa-gfs-forecast"
DEFAULT_TIME_END = "2023-12-31T23:59:59"

SURFACE_ALIASES = {
    "2m_temperature": ("temperature_2m", "2m_temperature"),
    "10m_u_component_of_wind": (
        "wind_u_10m",
        "u_component_of_wind_10m",
        "10m_u_component_of_wind",
        "u_wind_10m",
    ),
    "10m_v_component_of_wind": (
        "wind_v_10m",
        "v_component_of_wind_10m",
        "10m_v_component_of_wind",
        "v_wind_10m",
    ),
    "mean_sea_level_pressure": (
        "pressure_reduced_to_mean_sea_level",
        "mean_sea_level_pressure",
    ),
}

DEFAULT_LEAD_HOURS = (6, 12, 24, 48, 96, 192)
LEAD_COORD_CANDIDATES = (
    "prediction_timedelta",
    "lead_time",
    "lead_times",
    "step",
    "forecast_hour",
    "time_delta",
)
INIT_TIME_CANDIDATES = ("init_time", "forecast_reference_time", "time")
ENSEMBLE_MEMBER_CANDIDATES = ("ensemble_member", "number", "realization", "member")


def _import_dynamical_catalog():
    try:
        import dynamical_catalog  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install dynamical-catalog>=0.5.0 before running "
            "this downloader."
        ) from exc
    return dynamical_catalog


def _available_hours(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[h]").astype(int)
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float).round().astype(int)

    hours = []
    for value in values:
        text = str(value)
        try:
            hours.append(int(text))
            continue
        except ValueError:
            pass
        if text.endswith("h"):
            hours.append(int(text[:-1]))
            continue
        raise ValueError(f"Cannot interpret lead-time value {value!r} as hours.")
    return np.asarray(hours, dtype=int)


def _find_coord(ds, candidates: Iterable[str], *, required: bool) -> str | None:
    for name in candidates:
        if name in ds.dims or name in ds.coords:
            return name

    for name in list(ds.dims) + list(ds.coords):
        lower = name.lower()
        if any(candidate in lower for candidate in candidates):
            return name

    if required:
        raise SystemExit(
            f"Could not find any of these coordinates/dimensions: {list(candidates)}"
        )
    return None


def _select_lead_times(ds, lead_hours: list[int]):
    lead_coord = _find_coord(ds, LEAD_COORD_CANDIDATES, required=True)
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


def _select_init_time(ds, time_start: str | None, time_end: str | None):
    if time_start is None and time_end is None:
        print("No init-time range supplied; keeping the full available range.")
        return ds

    time_coord = _find_coord(ds, INIT_TIME_CANDIDATES, required=True)
    print(f"Filtering {time_coord} to {time_start or 'start'} through {time_end or 'end'}")
    return ds.sel({time_coord: slice(time_start, time_end)})


def _select_ensemble_members(ds, ensemble_members: list[int] | None):
    if ensemble_members is None:
        return ds

    member_coord = _find_coord(ds, ENSEMBLE_MEMBER_CANDIDATES, required=False)
    if member_coord is None:
        print("No ensemble-member coordinate found; skipping ensemble filtering.")
        return ds

    available = np.asarray(ds[member_coord].values).astype(int)
    selected = []
    missing = []
    source_values = ds[member_coord].values
    for member in ensemble_members:
        matches = np.where(available == member)[0]
        if len(matches) == 0:
            missing.append(member)
        else:
            selected.append(source_values[int(matches[0])])

    if missing:
        print(f"Warning: requested ensemble members not found: {missing}")
    if not selected:
        raise SystemExit(
            f"None of the requested ensemble members {ensemble_members} are available. "
            f"Available members: {available.tolist()}"
        )

    print(f"Filtering {member_coord} to ensemble members: {ensemble_members}")
    return ds.sel({member_coord: selected})


def _resolve_variables(ds, *, keep_source_names: bool) -> tuple[list[str], dict[str, str]]:
    selected = []
    rename = {}
    missing = {}

    for output_name, aliases in SURFACE_ALIASES.items():
        source_name = next((alias for alias in aliases if alias in ds.data_vars), None)
        if source_name is None:
            missing[output_name] = aliases
            continue

        selected.append(source_name)
        if not keep_source_names and source_name != output_name:
            rename[source_name] = output_name

    if missing:
        for output_name, aliases in missing.items():
            print(
                "Warning: missing "
                f"{output_name}; tried source names {list(aliases)}"
            )

    if not selected:
        raise SystemExit(
            "None of the default surface variables were found. "
            f"Available variables: {list(ds.data_vars)}"
        )

    return selected, rename


def _write_dataset(ds, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    ds = ds.drop_encoding()

    if suffix == ".zarr":
        print(f"Saving Zarr to {output_path}")
        ds.to_zarr(output_path, mode="w")
        return

    if suffix in {".nc", ".netcdf"}:
        print(f"Saving NetCDF to {output_path}")
        ds.to_netcdf(output_path, format="NETCDF4")
        return

    raise SystemExit(
        f"Unsupported output suffix {output_path.suffix!r}. "
        "Use .zarr, .nc, or .netcdf."
    )


def download_dynamical_forecast(
    output_path: Path,
    dataset_name: str,
    time_start: str | None,
    time_end: str | None,
    lead_hours: list[int],
    ensemble_members: list[int] | None,
    keep_source_names: bool,
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

    ds = _select_lead_times(ds, lead_hours)
    ds = _select_init_time(ds, time_start, time_end)
    ds = _select_ensemble_members(ds, ensemble_members)

    _write_dataset(ds, output_path)
    print("Done!")


download_gfs_dynamical = download_dynamical_forecast


def _parse_ensemble_members(value: str) -> list[int] | None:
    if value.lower() == "all":
        return None
    return [int(part) for part in value.split(",") if part]


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
        help="Optional init-time start, e.g. 2025-01-01T00. Default: dataset start.",
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
