# NOMADS Fast GFS Surface Downloads

This is an alternate download path for cases where the Dynamical/Icechunk route
is too slow. It follows NOAA NOMADS' fast-download procedure:

1. Read the small `.idx` inventory for a GRIB2 file.
2. Filter inventory rows for the wanted fields.
3. Use HTTP byte-range requests to download only those GRIB messages.

NOAA describes this procedure here:

https://nomads.ncep.noaa.gov/info.php?page=fastdownload

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
OUTPUT_DIR=data/nomads_gfs_fast \
download/download_nomads_gfs_fast.sh
```

Date range:

```bash
START_DATE=20260701 \
END_DATE=20260706 \
CYCLES="00 06 12 18" \
OUTPUT_DIR=data/nomads_gfs_fast \
download/download_nomads_gfs_fast.sh
```

## SLURM Usage

The sbatch wrapper has cluster-friendly defaults and activates the `pmlr` conda
environment.

```bash
DATE=20260706 CYCLES="00" \
sbatch -A pmlr_jobs -t 02:00 --export=ALL \
  download/sbatch_nomads_gfs_fast.sh
```

For a range:

```bash
START_DATE=20260701 END_DATE=20260706 CYCLES="00 06 12 18" \
sbatch -A pmlr_jobs -t 02:00 --export=ALL \
  download/sbatch_nomads_gfs_fast.sh
```

## Caveat

NOMADS is an operational server and generally exposes recent model cycles, not
the full historical range back to 2023. For older dates, use a NOAA archive
with the same `.idx` plus byte-range pattern if available.
