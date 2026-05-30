#!/usr/bin/env python
"""Check whether a nuScenes trainval tree matches the BEVFusion configs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_ROOT = (
    "data/nuscenes/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check raw nuScenes files and mmdet3d info pkls."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("NUSCENES_DATA_ROOT", DEFAULT_ROOT),
        help="nuScenes root directory used by the custom configs.",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes metadata version directory to check.",
    )
    parser.add_argument(
        "--extra-tag",
        default="nuscenes",
        help="Prefix used by tools/create_data.py for info files.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Return success when raw nuScenes files are present, even if info pkls are missing.",
    )
    parser.add_argument(
        "--require-mono3d",
        action="store_true",
        help="Also require optional mono3d COCO annotation json files.",
    )
    return parser.parse_args()


def check_paths(data_root: Path, version: str, extra_tag: str,
                require_mono3d: bool) -> list[tuple[str, Path, bool]]:
    camera_names = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    required = [
        ("root", data_root),
        ("metadata", data_root / version),
        ("samples", data_root / "samples"),
        ("sample lidar", data_root / "samples" / "LIDAR_TOP"),
        ("sweeps lidar", data_root / "sweeps" / "LIDAR_TOP"),
        ("maps", data_root / "maps"),
    ]
    required.extend(
        (f"sample camera {name}", data_root / "samples" / name)
        for name in camera_names
    )
    metadata_files = [
        "attribute.json",
        "calibrated_sensor.json",
        "category.json",
        "ego_pose.json",
        "instance.json",
        "log.json",
        "map.json",
        "sample.json",
        "sample_annotation.json",
        "sample_data.json",
        "scene.json",
        "sensor.json",
        "visibility.json",
    ]
    required.extend(
        (f"metadata {name}", data_root / version / name)
        for name in metadata_files
    )
    info_files = [
        f"{extra_tag}_infos_train.pkl",
        f"{extra_tag}_infos_val.pkl",
    ]
    if require_mono3d:
        info_files.extend([
            f"{extra_tag}_infos_train_mono3d.coco.json",
            f"{extra_tag}_infos_val_mono3d.coco.json",
        ])
    required.extend(
        (f"mmdet3d info {name}", data_root / name)
        for name in info_files
    )
    return [(label, path, path.exists()) for label, path in required]


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    rows = check_paths(data_root, args.version, args.extra_tag,
                       args.require_mono3d)
    max_label = max(len(label) for label, _, _ in rows)

    print(f"data_root: {data_root}")
    print(f"version:   {args.version}")
    print(f"extra_tag: {args.extra_tag}")
    print()
    for label, path, exists in rows:
        status = "OK" if exists else "MISSING"
        print(f"{status:7}  {label:<{max_label}}  {path}")

    raw_ok = all(
        exists
        for label, _, exists in rows
        if not label.startswith("mmdet3d info")
    )
    info_ok = all(
        exists
        for label, _, exists in rows
        if label.startswith("mmdet3d info")
    )
    print()
    print(f"raw_data_ready: {raw_ok}")
    print(f"info_pkls_ready: {info_ok}")
    if args.raw_only:
        return 0 if raw_ok else 1
    return 0 if raw_ok and info_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
