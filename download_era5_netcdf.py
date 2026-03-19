"""
Heavy inspiration from https://github.com/joeloskarsson/era5_data_handling/blob/main/download_era5.py
"""
#!/usr/bin/env python3
import argparse
from pathlib import Path
import xarray as xr


def download_era5_netcdf(
    output_path: Path,
    source: str,
    variables: list[str] | None,
    time_start: str,
    time_end: str,
):
    print(f"Opening {source}...")
    ds = xr.open_zarr(source, chunks=None)

    # Select variables
    if variables:
        ds = ds[variables]

    # Time slice
    ds = ds.sel(time=slice(time_start, time_end))

    if ds.time.size == 0:
        raise SystemExit("No data in specified time range.")

    print(f"Saving {ds.time.size} timesteps to {output_path}")

    # Save as NetCDF
    ds.to_netcdf(output_path)

    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output", type=Path)
    parser.add_argument("-v", "--variables", nargs="+")
    parser.add_argument("-s", "--time-start", required=True)
    parser.add_argument("-e", "--time-end", required=True)

    args = parser.parse_args()

    download_era5_netcdf(
        args.output,
        args.source,
        args.variables,
        args.time_start,
        args.time_end,
    )


if __name__ == "__main__":
    main()
