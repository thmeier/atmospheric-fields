#!/usr/bin/env python3
"""Convert a Zarr dataset to NetCDF."""

from __future__ import annotations

import argparse
import json
from numbers import Number
from pathlib import Path

import numpy as np


NETCDF_ATTR_TYPES = (str, Number, np.ndarray, np.number, list, tuple, bytes)


def _sanitize_attr_value(value):
    if value is None:
        return ""
    if isinstance(value, NETCDF_ATTR_TYPES):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _sanitize_attrs(obj):
    obj.attrs = {key: _sanitize_attr_value(value) for key, value in obj.attrs.items()}


def _sanitize_dataset_attrs(ds):
    _sanitize_attrs(ds)
    for variable in ds.variables.values():
        _sanitize_attrs(variable)
    return ds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_zarr", type=Path, help="Input .zarr dataset")
    parser.add_argument("output_nc", type=Path, help="Output .nc/.netcdf file")
    parser.add_argument(
        "--chunks",
        default=None,
        help=(
            "Optional xarray chunk mapping, e.g. init_time=1,lead_time=-1. "
            "By default, use the stored Zarr chunks."
        ),
    )
    args = parser.parse_args()

    chunks = None
    if args.chunks:
        chunks = {}
        for item in args.chunks.split(","):
            dim, value = item.split("=", 1)
            chunks[dim] = int(value)

    import xarray as xr

    ds = xr.open_zarr(args.input_zarr, chunks=chunks)
    ds = _sanitize_dataset_attrs(ds)
    args.output_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.output_nc, format="NETCDF4")


if __name__ == "__main__":
    main()
