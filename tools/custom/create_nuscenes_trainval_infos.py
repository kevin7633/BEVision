#!/usr/bin/env python
"""Create mmdet3d nuScenes trainval infos without requiring test split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mmengine.registry import init_default_scope

from tools.dataset_converters import nuscenes_converter
from tools.dataset_converters.create_gt_database import (
    create_groundtruth_database,
)
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-path",
        default=os.environ.get(
            "NUSCENES_DATA_ROOT",
            "data/nuscenes/",
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get(
            "NUSCENES_DATA_ROOT",
            "data/nuscenes/",
        ),
    )
    parser.add_argument("--extra-tag", default="nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument(
        "--skip-create-infos",
        action="store_true",
        help="Reuse existing raw nuscenes_infos_train/val.pkl files.",
    )
    parser.add_argument(
        "--skip-gt-database",
        action="store_true",
        help="Skip optional GT database creation.",
    )
    parser.add_argument(
        "--skip-cam-instances",
        action="store_true",
        help="Skip slow per-camera 2D instance annotations for frame-based 3D detection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_path = str(Path(args.root_path).expanduser().resolve())
    out_dir = str(Path(args.out_dir).expanduser().resolve())

    init_default_scope("mmdet3d")
    print(f"root_path={root_path}")
    print(f"out_dir={out_dir}")
    print(f"extra_tag={args.extra_tag}")
    print(f"version={args.version}")
    print(f"max_sweeps={args.max_sweeps}")

    if not args.skip_create_infos:
        nuscenes_converter.create_nuscenes_infos(
            root_path,
            args.extra_tag,
            version=args.version,
            max_sweeps=args.max_sweeps,
        )
    train_info = str(Path(out_dir) / f"{args.extra_tag}_infos_train.pkl")
    val_info = str(Path(out_dir) / f"{args.extra_tag}_infos_val.pkl")
    update_pkl_infos(
        "nuscenes",
        out_dir=out_dir,
        pkl_path=train_info,
        skip_cam_instances=args.skip_cam_instances,
    )
    update_pkl_infos(
        "nuscenes",
        out_dir=out_dir,
        pkl_path=val_info,
        skip_cam_instances=args.skip_cam_instances,
    )
    if not args.skip_gt_database:
        create_groundtruth_database(
            "NuScenesDataset",
            root_path,
            args.extra_tag,
            f"{args.extra_tag}_infos_train.pkl",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
