#!/usr/bin/env python
"""Collect reliability gate statistics from MMEngine text or JSON logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean


KEYS = [
    "reliability/gate_mean",
    "reliability/gate_min",
    "reliability/gate_max",
    "reliability/image_reliability_mean",
    "reliability/lidar_reliability_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="MMEngine log files.")
    parser.add_argument("--out", help="Optional CSV output path.")
    return parser.parse_args()


def values_from_json(line: str) -> dict[str, float]:
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        return {}
    values = {}
    for key in KEYS:
        if key in item:
            try:
                values[key] = float(item[key])
            except (TypeError, ValueError):
                pass
    return values


def values_from_text(line: str) -> dict[str, float]:
    values = {}
    for key in KEYS:
        pattern = rf"{re.escape(key)}[:=]\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            values[key] = float(match.group(1))
    return values


def collect(path: Path) -> dict[str, object]:
    buckets: dict[str, list[float]] = {key: [] for key in KEYS}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            values = values_from_json(line)
            if not values:
                values = values_from_text(line)
            for key, value in values.items():
                buckets[key].append(value)

    row: dict[str, object] = {"log": str(path), "records": 0}
    for key, items in buckets.items():
        row["records"] = max(int(row["records"]), len(items))
        short = key.split("/", 1)[1]
        row[f"{short}_mean"] = mean(items) if items else ""
        row[f"{short}_last"] = items[-1] if items else ""
    return row


def main() -> int:
    args = parse_args()
    rows = [collect(Path(log).expanduser().resolve()) for log in args.logs]
    fieldnames = ["log", "records"]
    for key in KEYS:
        short = key.split("/", 1)[1]
        fieldnames.extend([f"{short}_mean", f"{short}_last"])

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote: {out_path}")
    else:
        print(",".join(fieldnames))
        for row in rows:
            print(",".join(str(row.get(name, "")) for name in fieldnames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
