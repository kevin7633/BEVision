#!/usr/bin/env python3
"""Compare clean and corrupted BEVFusion pipeline outputs for one sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import DATASETS


def init_project(cfg: Config, repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    else:
        import projects.BEVFusion.bevfusion  # noqa: F401
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))


def build_dataset(config_path: str, repo_root: Path):
    cfg = Config.fromfile(config_path)
    init_project(cfg, repo_root)
    return DATASETS.build(cfg.test_dataloader.dataset)


def get_img_tensor(sample: dict) -> torch.Tensor:
    imgs = sample['inputs']['img']
    return imgs[0] if isinstance(imgs, list) else imgs


def get_points_tensor(sample: dict) -> torch.Tensor:
    points = sample['inputs']['points']
    points = points[0] if isinstance(points, list) else points
    return points if isinstance(points, torch.Tensor) else points.tensor


def chw_to_rgb(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_views(clean_imgs: torch.Tensor, corr_imgs: torch.Tensor,
               out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for view_idx in range(clean_imgs.shape[0]):
        clean = chw_to_rgb(clean_imgs[view_idx])
        corr = chw_to_rgb(corr_imgs[view_idx])
        diff = np.abs(clean.astype(np.float32) - corr.astype(np.float32))
        diff = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
        canvas = np.concatenate([clean, corr, diff], axis=1)
        cv2.imwrite(str(out_dir / f'{prefix}_view{view_idx}_clean_corr_diff.jpg'),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def summarize(clean_sample: dict, corr_sample: dict) -> dict:
    clean_imgs = get_img_tensor(clean_sample)
    corr_imgs = get_img_tensor(corr_sample)
    clean_points = get_points_tensor(clean_sample)
    corr_points = get_points_tensor(corr_sample)

    img_abs_diff = (clean_imgs.float() - corr_imgs.float()).abs()
    summary = {
        'sample_idx': clean_sample['data_samples'].metainfo.get('sample_idx'),
        'image': {
            'clean_shape': list(clean_imgs.shape),
            'corrupted_shape': list(corr_imgs.shape),
            'clean_min': float(clean_imgs.min()),
            'clean_max': float(clean_imgs.max()),
            'clean_mean': float(clean_imgs.float().mean()),
            'corrupted_min': float(corr_imgs.min()),
            'corrupted_max': float(corr_imgs.max()),
            'corrupted_mean': float(corr_imgs.float().mean()),
            'mean_abs_diff': float(img_abs_diff.mean()),
            'max_abs_diff': float(img_abs_diff.max()),
        },
        'points': {
            'clean_shape': list(clean_points.shape),
            'corrupted_shape': list(corr_points.shape),
            'clean_count': int(clean_points.shape[0]),
            'corrupted_count': int(corr_points.shape[0]),
            'count_delta': int(corr_points.shape[0] - clean_points.shape[0]),
            'count_ratio': float(corr_points.shape[0] / max(1, clean_points.shape[0])),
            'clean_xyz_mean': clean_points[:, :3].float().mean(0).tolist(),
            'corrupted_xyz_mean': corr_points[:, :3].float().mean(0).tolist(),
        }
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-config', required=True)
    parser.add_argument('--corrupted-config', required=True)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--repo-root', default='/workspace/mmdetection3d')
    parser.add_argument('--out-dir',
                        default='/workspace/analysis_results/qualitative/corruption_verification')
    parser.add_argument('--summary-json',
                        default='/workspace/analysis_results/corruption_verification_summary.json')
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    clean_dataset = build_dataset(args.clean_config, repo_root)
    corr_dataset = build_dataset(args.corrupted_config, repo_root)
    clean_sample = clean_dataset[args.sample_index]
    corr_sample = corr_dataset[args.sample_index]

    summary = summarize(clean_sample, corr_sample)
    out_dir = Path(args.out_dir)
    save_views(
        get_img_tensor(clean_sample),
        get_img_tensor(corr_sample),
        out_dir,
        prefix=f'sample{args.sample_index}')

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
