# Fast GFS Surface Downloads

This is an alternate download path for cases where the Dynamical/Icechunk route
is too slow. It follows NOAA NOMADS' fast-download procedure, but defaults to
the NOAA Open Data S3 GFS bucket because that backend exposes historical GFS
GRIB2 plus `.idx` files with the same path layout:

1. Read the small `.idx` inventory for a GRIB2 file.
2. Filter inventory rows for the wanted fields.
3. Use HTTP byte-range requests to download only those GRIB messages.

NOAA describes this procedure here:

https://nomads.ncep.noaa.gov/info.php?page=fastdownload

Default base URL:

```text
https://noaa-gfs-bdp-pds.s3.amazonaws.com
```

This points to the AWS Open Data `noaa-gfs-bdp-pds` bucket. The AWS registry
entry documents the bucket and no-sign-request access:

https://registry.opendata.aws/noaa-gfs-bdp-pds/

The Dynamical archive is not a drop-in replacement for this fast path. The
public Dynamical Catalog exposes GFS/GEFS as Icechunk/Zarr datasets, not as raw
GRIB2 files plus `.idx` inventories. That is useful for xarray access, but it
does not give us GRIB-message byte ranges for server-side field selection.

The downloaded GFS files contain only these four surface fields:

| Repo field | GFS inventory row |
| --- | --- |
| `mean_sea_level_pressure` | `PRMSL:mean sea level` |
| `2m_temperature` | `TMP:2 m above ground` |
| `10m_u_component_of_wind` | `UGRD:10 m above ground` |
| `10m_v_component_of_wind` | `VGRD:10 m above ground` |

Default lead hours are `6 12 24 48 96 192`.

## Local Usage

Single date:

```bash
DATE=20260706 \
CYCLES="00" \
OUTPUT_DIR=data/gfs_fast \
download/download_nomads_gfs_fast.sh
```

Date range:

```bash
START_DATE=20260701 \
END_DATE=20260706 \
CYCLES="00 06 12 18" \
OUTPUT_DIR=data/gfs_fast \
download/download_nomads_gfs_fast.sh
```

## SLURM Usage

The sbatch wrapper has cluster-friendly defaults and activates the `pmlr` conda
environment.

```bash
sbatch -A pmlr_jobs -t 02:00 download/sbatch_nomads_gfs_fast.sh
```

The GFS sbatch wrapper defaults to:

- `START_DATE=20210501`
- `END_DATE=20231231`
- `CYCLES="00 06 12 18"`
- `LEAD_HOURS="6 12 24 48 96 192"`
- `OUTPUT_DIR=/cluster/courses/pmlr/teams/team07/data/gfs_fast`

To turn the downloaded GFS GRIB2 files into native selected-field Zarr,
WeatherBench2-regridded Zarr, and final NetCDF:

```bash
MODEL=gfs sbatch -A pmlr_jobs -t 04:00 --export=ALL \
  download/sbatch_process_fast_surface_forecasts.sh
```

## GEFS

GEFS uses the same byte-range approach against the NOAA GEFS S3 archive:

```text
https://noaa-gefs-pds.s3.amazonaws.com
```

Control member from `2020-10-01` through the end of 2023:

```bash
sbatch -A pmlr_jobs -t 02:00 download/sbatch_nomads_gefs_fast.sh
```

The GEFS sbatch wrapper defaults to:

- `START_DATE=20201001`
- `END_DATE=20231231`
- `CYCLES="00"`
- `MEMBERS="0"`
- `LEAD_HOURS="6 12 24 48 96 192"`
- `OUTPUT_DIR=/cluster/courses/pmlr/teams/team07/data/gefs_fast`

All GEFS members:

```bash
MEMBERS=all \
sbatch -A pmlr_jobs -t 02:00 --export=ALL \
  download/sbatch_nomads_gefs_fast.sh
```

To turn the downloaded GEFS GRIB2 files into native selected-field Zarr,
WeatherBench2-regridded Zarr, and final NetCDF:

```bash
MODEL=gefs sbatch -A pmlr_jobs -t 04:00 --export=ALL \
  download/sbatch_process_fast_surface_forecasts.sh
```

The processing wrapper uses `cfgrib`/`xarray` to make a native selected-field
Zarr, calls WeatherBench2's `scripts/regrid.py` with conservative regridding to
`240x121`, then converts the regridded Zarr to NetCDF by default.

## Backend Choice

Use the default NOAA S3 backend for historical dates such as 2023. To use the
operational NOMADS server for recent cycles, override:

```bash
BASE_URL=https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod \
DATE=20260706 \
CYCLES="00" \
download/download_nomads_gfs_fast.sh
```
