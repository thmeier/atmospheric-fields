"""
Heavy inspiration from https://github.com/joeloskarsson/era5_data_handling/blob/main/download_era5.py
"""
#!/usr/bin/env python3
import argparse
from pathlib import Path
import xarray as xr
import numpy as np


def download_era5_netcdf(
    output_path: Path,
    source: str,
    variables: list[str] | None,
    time_start: str,
    time_end: str,
):
    print(f"Opening {source}...")
    ds = xr.open_zarr(source, chunks="auto", storage_options={'token': 'anon'})

    # Select variables
    if variables:
        ds = ds[variables]

    # Time slice
    ds = ds.sel(time=slice(time_start, time_end))

    if ds.time.size == 0:
        raise SystemExit("No data in specified time range.")

    # Lead time filtering
    lead_time_dims = [d for d in ds.dims if "timedelta" in d or "lead_time" in d]
    if lead_time_dims:
        lt_dim = lead_time_dims[0]
        print(f"Detected lead time dimension: {lt_dim}")

        target_hours = [6, 12, 24, 48, 96, 192]
        target_lt = np.array(target_hours, dtype="timedelta64[h]").astype(
            "timedelta64[ns]"
        )

        available_lt = ds[lt_dim].values
        valid_lt = [lt for lt in target_lt if lt in available_lt]
        missing_lt = [lt for lt in target_lt if lt not in available_lt]

        if missing_lt:
            missing_hours = [
                int(lt.astype("timedelta64[h]").astype(int)) for lt in missing_lt
            ]
            print(f"Warning: Requested lead times {missing_hours}h not found in dataset.")

        if not valid_lt:
            print(f"Error: None of the requested lead times {target_hours}h are available.")
            if len(available_lt) > 0:
                print(f"Available lead times (first 10): {available_lt[:10]}")
            raise SystemExit("No matching lead times found.")

        valid_hours = [int(lt.astype("timedelta64[h]").astype(int)) for lt in valid_lt]
        print(f"Filtering for lead times: {valid_hours}h")
        ds = ds.sel({lt_dim: valid_lt})
    else:
        print("No lead time dimension detected. Proceeding with full dataset.")

    print(f"Saving {ds.time.size} timesteps to {output_path}")

    # Save as NetCDF
    ds.to_netcdf(output_path, format="NETCDF4")

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
