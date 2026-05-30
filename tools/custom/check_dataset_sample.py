#!/usr/bin/env python
"""Build a configured dataset and fetch one sample for smoke testing."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Config file path.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataloader split to build.",
    )
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def import_custom_modules(cfg: Config) -> None:
    custom_imports = cfg.get("custom_imports", None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)


def unwrap_dataset_cfg(dataset_cfg):
    cfg = deepcopy(dataset_cfg)
    # The BEVFusion train config wraps NuScenesDataset in CBGSDataset.
    while isinstance(cfg, dict) and "dataset" in cfg and "type" in cfg:
        if cfg["type"] in {"CBGSDataset", "RepeatDataset", "ClassBalancedDataset"}:
            cfg = cfg["dataset"]
        else:
            break
    return cfg


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = Config.fromfile(cfg_path)
    import_custom_modules(cfg)
    register_all_modules(init_default_scope=True)

    dataloader_cfg = cfg[f"{args.split}_dataloader"]
    dataset_cfg = unwrap_dataset_cfg(dataloader_cfg["dataset"])
    dataset = DATASETS.build(dataset_cfg)
    print(f"config={cfg_path}")
    print(f"split={args.split}")
    print(f"dataset={dataset.__class__.__name__}")
    print(f"length={len(dataset)}")

    item = dataset[args.index]
    print(f"sample_index={args.index}")
    print(f"sample_type={type(item).__name__}")
    if isinstance(item, dict):
        print(f"sample_keys={sorted(item.keys())}")
        inputs = item.get("inputs", {})
        if isinstance(inputs, dict):
            print(f"input_keys={sorted(inputs.keys())}")
            for key, value in inputs.items():
                shape = getattr(value, "shape", None)
                dtype = getattr(value, "dtype", None)
                print(f"input[{key}] shape={shape} dtype={dtype}")
        data_samples = item.get("data_samples", None)
        if data_samples is not None:
            print(f"data_sample_type={type(data_samples).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
