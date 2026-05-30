#!/usr/bin/env python
"""Load a config and optionally build the model without touching data."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check a mmdet3d config before launching jobs."
    )
    parser.add_argument("config", help="Path to config file.")
    parser.add_argument(
        "--build-model",
        action="store_true",
        help="Also build the model module. This does not load dataset samples.",
    )
    parser.add_argument(
        "--skip-init-cfg",
        action="store_true",
        help="Drop pretrained init_cfg entries before --build-model.",
    )
    return parser.parse_args()


def import_custom_modules(cfg: Config) -> None:
    custom_imports = cfg.get("custom_imports", None)
    if custom_imports is None:
        return
    import_modules_from_strings(**custom_imports)


def strip_init_cfg(item) -> None:
    if isinstance(item, dict):
        item.pop("init_cfg", None)
        for value in item.values():
            strip_init_cfg(value)
    elif isinstance(item, list):
        for value in item:
            strip_init_cfg(value)


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = Config.fromfile(cfg_path)
    import_custom_modules(cfg)
    register_all_modules(init_default_scope=True)

    print(f"config:   {cfg_path}")
    print(f"work_dir: {cfg.get('work_dir', '<unset>')}")
    print(f"model:    {cfg.model.get('type', '<unknown>')}")
    print(f"data:     {cfg.get('data_root', '<unset>')}")

    reliability = cfg.model.get("reliability_fusion_layer", None)
    if reliability is None:
        print("reliability_fusion_layer: disabled")
    else:
        print("reliability_fusion_layer: enabled")
        print(f"  type: {reliability.get('type')}")
        print(f"  gate_mode: {reliability.get('gate_mode')}")
        print(f"  gate_formula: {reliability.get('gate_formula')}")

    if args.build_model:
        if args.skip_init_cfg:
            strip_init_cfg(cfg.model)
        model = MODELS.build(cfg.model)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"model_build: OK ({num_params:,} parameters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
