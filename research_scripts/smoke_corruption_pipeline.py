#!/usr/bin/env python3
"""Build a BEVFusion dataloader and inspect one corrupted mini sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import DATASETS, TRANSFORMS
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--repo-root', default='/workspace/mmdetection3d')
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    sys.path.insert(0, str(repo_root))
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    else:
        import projects.BEVFusion.bevfusion  # noqa: F401
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    dataset_cfg = cfg.test_dataloader.dataset
    dataset = DATASETS.build(dataset_cfg)
    sample = dataset[0]
    batch = pseudo_collate([sample])

    inputs = batch['inputs']
    imgs = inputs.get('img')
    points = inputs.get('points')
    print(f'config: {args.config}')
    print(f'dataset_len: {len(dataset)}')
    if imgs is not None:
        print(f'img_type: {type(imgs)}')
        if isinstance(imgs, list):
            img_tensor = imgs[0]
        else:
            img_tensor = imgs
        print(f'img_shape: {tuple(img_tensor.shape)}')
        print(f'img_min_max: {float(img_tensor.min()):.3f}, {float(img_tensor.max()):.3f}')
    if points is not None:
        first_points = points[0] if isinstance(points, list) else points
        print(f'points_type: {type(first_points)}')
        if not isinstance(first_points, torch.Tensor):
            first_points = first_points.tensor
        print(f'points_shape: {tuple(first_points.shape)}')
        print(f'points_xyz_mean: {first_points[:, :3].mean(0).tolist()}')


if __name__ == '__main__':
    main()
