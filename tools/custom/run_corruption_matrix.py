#!/usr/bin/env python
"""Print or run BEVFusion corruption evaluation commands."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CONDITIONS = {
    "clean": ("none", "none"),
    "image_brightness_down": ("brightness_down", "none"),
    "image_blur": ("blur", "none"),
    "image_dropout": ("dropout_camera", "none"),
    "lidar_point_dropout": ("none", "point_dropout"),
    "image_lidar_degradation": ("all", "point_dropout"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the same checkpoint under clean/corrupt settings."
    )
    parser.add_argument(
        "--config",
        default="configs/custom/bevfusion_corruption_runtime_eval.py",
        help="Config that reads BEVFUSION_* corruption environment variables.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument(
        "--work-root",
        default=os.environ.get(
            "BEVFUSION_WORK_ROOT",
            "work_dirs",
        ),
        help="Root directory for condition-specific eval outputs.",
    )
    parser.add_argument(
        "--severity",
        default="moderate",
        choices=["mild", "moderate", "severe"],
        help="Corruption severity used by the runtime eval config.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run tools/test.py.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS.keys()),
        choices=list(CONDITIONS.keys()),
        help="Subset of conditions to evaluate.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run commands sequentially. Without this flag, only print them.",
    )
    return parser.parse_args()


def shell_env_prefix(env_items: dict[str, str]) -> str:
    return " ".join(
        f"{key}={shlex.quote(value)}" for key, value in env_items.items()
    )


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.work_root).expanduser().resolve()
    commands = []

    for condition in args.conditions:
        camera, lidar = CONDITIONS[condition]
        work_dir = root / f"eval_{stamp}_{condition}_{args.severity}"
        env_items = {
            "BEVFUSION_CAMERA_CORRUPTION": camera,
            "BEVFUSION_LIDAR_CORRUPTION": lidar,
            "BEVFUSION_CORRUPTION_SEVERITY": args.severity,
        }
        cmd = [
            args.python,
            "tools/test.py",
            args.config,
            args.checkpoint,
            "--work-dir",
            str(work_dir),
        ]
        commands.append((condition, env_items, cmd))

    for condition, env_items, cmd in commands:
        rendered = f"{shell_env_prefix(env_items)} {' '.join(shlex.quote(x) for x in cmd)}"
        print(f"\n# {condition}")
        print(rendered)

    if not args.run:
        return 0

    for condition, env_items, cmd in commands:
        print(f"\n[run] {condition}", flush=True)
        env = os.environ.copy()
        env.update(env_items)
        subprocess.run(cmd, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
