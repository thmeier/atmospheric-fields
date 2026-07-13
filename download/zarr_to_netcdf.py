#!/usr/bin/env python3
"""Convert a Zarr dataset to NetCDF."""

from __future__ import annotations

import argparse
from pathlib import Path


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
    args.output_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.output_nc, format="NETCDF4")


if __name__ == "__main__":
    main()
