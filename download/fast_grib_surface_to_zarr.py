#!/usr/bin/env python3
"""Convert fast-downloaded GFS/GEFS surface GRIB2 files to native-grid Zarr."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import xarray as xr


DEFAULT_LEAD_HOURS = (6, 12, 24, 48, 96, 192)
FIELD_SPECS = {
    "mean_sea_level_pressure": {"shortName": "prmsl"},
    "2m_temperature": {"shortName": "2t"},
    "10m_u_component_of_wind": {"shortName": "10u"},
    "10m_v_component_of_wind": {"shortName": "10v"},
}


def _parse_yyyymmdd(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y%m%d").date()


def _date_range(start: dt.date, end: dt.date):
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def _init_time(date: dt.date, cycle: str) -> np.datetime64:
    return np.datetime64(f"{date.strftime('%Y-%m-%d')}T{cycle}:00:00")


def _member_filename(member: str) -> str:
    normalized = member.lower()
    if normalized in {"0", "c00", "ctl", "control", "gec00"}:
        return "gec00"
    if normalized.startswith("gep"):
        return normalized
    if normalized.startswith("p"):
        return f"gep{int(normalized[1:]):02d}"
    return f"gep{int(normalized):02d}"


def _parse_members(values: list[str]) -> list[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
    return [_member_filename(value) for value in values]


def _grib_path(input_dir: Path, model: str, date: dt.date, cycle: str, lead: int, member: str | None):
    ymd = date.strftime("%Y%m%d")
    if model == "gfs":
        return (
            input_dir
            / f"gfs.{ymd}"
            / cycle
            / f"gfs.t{cycle}z.surface_0p25.f{lead:03d}.grib2"
        )

    if member is None:
        raise ValueError("GEFS requires a member name")
    return (
        input_dir
        / f"gefs.{ymd}"
        / cycle
        / member
        / f"{member}.t{cycle}z.surface_0p25.f{lead:03d}.grib2"
    )


def _open_field(path: Path, output_name: str):
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": FIELD_SPECS[output_name],
        },
    )
    if len(ds.data_vars) != 1:
        raise RuntimeError(f"Expected one variable for {output_name} in {path}, got {list(ds.data_vars)}")
    source_name = next(iter(ds.data_vars))
    da = ds[source_name].squeeze(drop=True).reset_coords(drop=True)
    return da.rename(output_name)


def _open_grib_dataset(path: Path) -> xr.Dataset:
    data_vars = {}
    for output_name in FIELD_SPECS:
        data_vars[output_name] = _open_field(path, output_name)
    return xr.Dataset(data_vars)


def _load_gfs_init(input_dir: Path, date: dt.date, cycle: str, lead_hours: list[int]):
    lead_datasets = []
    for lead in lead_hours:
        path = _grib_path(input_dir, "gfs", date, cycle, lead, None)
        if not path.exists():
            raise FileNotFoundError(path)
        ds = _open_grib_dataset(path).expand_dims(
            lead_time=[np.timedelta64(lead, "h")]
        )
        lead_datasets.append(ds)

    ds = xr.concat(lead_datasets, dim="lead_time")
    return ds.expand_dims(init_time=[_init_time(date, cycle)])


def _load_gefs_init(
    input_dir: Path,
    date: dt.date,
    cycle: str,
    lead_hours: list[int],
    members: list[str],
):
    member_datasets = []
    for member in members:
        lead_datasets = []
        for lead in lead_hours:
            path = _grib_path(input_dir, "gefs", date, cycle, lead, member)
            if not path.exists():
                raise FileNotFoundError(path)
            ds = _open_grib_dataset(path).expand_dims(
                lead_time=[np.timedelta64(lead, "h")]
            )
            lead_datasets.append(ds)
        member_ds = xr.concat(lead_datasets, dim="lead_time").expand_dims(
            ensemble_member=[member]
        )
        member_datasets.append(member_ds)

    ds = xr.concat(member_datasets, dim="ensemble_member")
    return ds.expand_dims(init_time=[_init_time(date, cycle)])


def _write_incremental(ds: xr.Dataset, output_zarr: Path, *, first: bool):
    ds = ds.drop_encoding()
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    if first:
        ds.to_zarr(output_zarr, mode="w")
    else:
        ds.to_zarr(output_zarr, mode="a", append_dim="init_time")


def convert_to_zarr(
    model: str,
    input_dir: Path,
    output_zarr: Path,
    start_date: dt.date,
    end_date: dt.date,
    cycles: list[str],
    lead_hours: list[int],
    members: list[str],
    overwrite: bool,
):
    if output_zarr.exists():
        if overwrite:
            shutil.rmtree(output_zarr)
        else:
            raise SystemExit(f"Output exists: {output_zarr}. Pass --overwrite to replace it.")

    first = True
    for date in _date_range(start_date, end_date):
        for cycle in cycles:
            print(f"Processing {model} {date.strftime('%Y%m%d')} {cycle}Z", flush=True)
            if model == "gfs":
                ds = _load_gfs_init(input_dir, date, cycle, lead_hours)
            else:
                ds = _load_gefs_init(input_dir, date, cycle, lead_hours, members)
            _write_incremental(ds, output_zarr, first=first)
            first = False

    print(f"Wrote {output_zarr}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("gfs", "gefs"))
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_zarr", type=Path)
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--cycles", nargs="+", default=["00"])
    parser.add_argument("--lead-hours", nargs="+", type=int, default=list(DEFAULT_LEAD_HOURS))
    parser.add_argument("--members", nargs="+", default=["0"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    convert_to_zarr(
        model=args.model,
        input_dir=args.input_dir,
        output_zarr=args.output_zarr,
        start_date=_parse_yyyymmdd(args.start_date),
        end_date=_parse_yyyymmdd(args.end_date),
        cycles=args.cycles,
        lead_hours=args.lead_hours,
        members=_parse_members(args.members),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
