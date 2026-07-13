#!/usr/bin/env python3
"""Fast selective GEFS GRIB2 downloads using .idx byte ranges."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://noaa-gefs-pds.s3.amazonaws.com"
DEFAULT_LEAD_HOURS = (6, 12, 24, 48, 96, 192)
FIELD_PATTERN = re.compile(
    r":(?:PRMSL:mean sea level|TMP:2 m above ground|"
    r"UGRD:10 m above ground|VGRD:10 m above ground):"
)


@dataclass(frozen=True)
class InventoryRow:
    message_number: int
    start: int
    end: int | None
    text: str


def _parse_yyyymmdd(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y%m%d").date()


def _date_range(start: dt.date, end: dt.date):
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


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


def _gefs_urls(base_url: str, date: dt.date, cycle: str, lead_hour: int, member: str):
    ymd = date.strftime("%Y%m%d")
    fff = f"{lead_hour:03d}"
    filename = f"{member}.t{cycle}z.pgrb2s.0p25.f{fff}"
    root = f"{base_url}/gefs.{ymd}/{cycle}/atmos/pgrb2sp25/{filename}"
    return root, f"{root}.idx"


def _read_text_url(url: str, timeout: int, retries: int) -> str:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed to read {url}: {last_error}") from last_error


def _parse_inventory(text: str) -> list[InventoryRow]:
    rows = []
    raw_rows = []
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        raw_rows.append((int(parts[0]), int(parts[1]), line))

    for idx, (message_number, start, line) in enumerate(raw_rows):
        end = raw_rows[idx + 1][1] - 1 if idx + 1 < len(raw_rows) else None
        rows.append(InventoryRow(message_number, start, end, line))
    return rows


def _selected_rows(index_text: str) -> list[InventoryRow]:
    rows = [row for row in _parse_inventory(index_text) if FIELD_PATTERN.search(row.text)]
    if len(rows) != 4:
        found = "\n".join(row.text for row in rows) or "(none)"
        raise RuntimeError(f"Expected 4 selected GEFS surface fields, found {len(rows)}:\n{found}")
    return rows


def _download_range(url: str, row: InventoryRow, timeout: int, retries: int) -> bytes:
    range_header = f"bytes={row.start}-" if row.end is None else f"bytes={row.start}-{row.end}"
    request = Request(url, headers={"Range": range_header})

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed range {range_header} from {url}: {last_error}") from last_error


def _download_one(
    grib_url: str,
    idx_url: str,
    output_path: Path,
    timeout: int,
    retries: int,
    overwrite: bool,
):
    if output_path.exists() and not overwrite:
        print(f"Exists, skipping: {output_path}", flush=True)
        return

    print(f"Reading index: {idx_url}", flush=True)
    index_text = _read_text_url(idx_url, timeout=timeout, retries=retries)
    rows = _selected_rows(index_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        for row in rows:
            print(f"  {row.text}", flush=True)
            handle.write(_download_range(grib_url, row, timeout=timeout, retries=retries))
    tmp_path.replace(output_path)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MiB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory for selected GEFS GRIB2 files")
    parser.add_argument("--date", help="Single cycle date, YYYYMMDD")
    parser.add_argument("--start-date", help="Start date, YYYYMMDD")
    parser.add_argument("--end-date", help="End date, YYYYMMDD")
    parser.add_argument(
        "--cycles",
        nargs="+",
        default=["00"],
        help="GEFS cycle hours to download.",
    )
    parser.add_argument(
        "--lead-hours",
        nargs="+",
        type=int,
        default=list(DEFAULT_LEAD_HOURS),
        help="Forecast lead hours to download.",
    )
    parser.add_argument(
        "--members",
        nargs="+",
        default=["0"],
        help="Members: 0/c00/control, 1..30, p01..p30, gep01..gep30, or all.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.date:
        start_date = end_date = _parse_yyyymmdd(args.date)
    elif args.start_date and args.end_date:
        start_date = _parse_yyyymmdd(args.start_date)
        end_date = _parse_yyyymmdd(args.end_date)
    else:
        raise SystemExit("Pass either --date YYYYMMDD or --start-date/--end-date.")

    for cycle in args.cycles:
        if cycle not in {"00", "06", "12", "18"}:
            raise SystemExit(f"Unsupported GEFS cycle {cycle!r}; use 00, 06, 12, or 18.")

    members = _parse_members(args.members)

    for date in _date_range(start_date, end_date):
        ymd = date.strftime("%Y%m%d")
        for cycle in args.cycles:
            for member in members:
                for lead_hour in args.lead_hours:
                    grib_url, idx_url = _gefs_urls(args.base_url, date, cycle, lead_hour, member)
                    output_path = (
                        args.output_dir
                        / f"gefs.{ymd}"
                        / cycle
                        / member
                        / f"{member}.t{cycle}z.surface_0p25.f{lead_hour:03d}.grib2"
                    )
                    _download_one(
                        grib_url,
                        idx_url,
                        output_path,
                        timeout=args.timeout,
                        retries=args.retries,
                        overwrite=args.overwrite,
                    )


if __name__ == "__main__":
    main()
