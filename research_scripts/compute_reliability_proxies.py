#!/usr/bin/env python3
"""Compute camera/LiDAR reliability proxy scores for nuScenes-mini samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import DATASETS


DEFAULT_EXPERIMENTS = [
    ('bevfusion_mini_clean_baseline', 'none', 'none', 'none', 'none'),
    ('bevfusion_mini_camera_fog_mild', 'camera_fog', 'mild', 'fog', 'none'),
    ('bevfusion_mini_camera_fog_moderate', 'camera_fog', 'moderate', 'fog',
     'none'),
    ('bevfusion_mini_camera_fog_severe', 'camera_fog', 'severe', 'fog',
     'none'),
    ('bevfusion_mini_lidar_dropout_mild', 'lidar_random_dropout', 'mild',
     'none', 'random_dropout'),
    ('bevfusion_mini_lidar_dropout_moderate', 'lidar_random_dropout',
     'moderate', 'none', 'random_dropout'),
    ('bevfusion_mini_lidar_dropout_severe', 'lidar_random_dropout', 'severe',
     'none', 'random_dropout'),
    ('bevfusion_mini_camera_lidar_all_mild', 'camera_lidar_all', 'mild', 'all',
     'all'),
    ('bevfusion_mini_camera_lidar_all_moderate', 'camera_lidar_all',
     'moderate', 'all', 'all'),
    ('bevfusion_mini_camera_lidar_all_severe', 'camera_lidar_all', 'severe',
     'all', 'all'),
]


def init_project(cfg: Config, repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    else:
        import projects.BEVFusion.bevfusion  # noqa: F401
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))


def build_dataset(config_path: Path, repo_root: Path, exp_name: str,
                  cam_corr: str, lidar_corr: str, severity: str):
    os.environ['BEVFUSION_EXP_NAME'] = exp_name
    os.environ['BEVFUSION_CAMERA_CORRUPTION'] = cam_corr
    os.environ['BEVFUSION_LIDAR_CORRUPTION'] = lidar_corr
    os.environ['BEVFUSION_CORRUPTION_SEVERITY'] = severity
    cfg = Config.fromfile(str(config_path))
    init_project(cfg, repo_root)
    return DATASETS.build(cfg.test_dataloader.dataset)


def get_img_tensor(sample: dict) -> torch.Tensor:
    imgs = sample['inputs']['img']
    return imgs[0] if isinstance(imgs, list) else imgs


def get_points_tensor(sample: dict) -> torch.Tensor:
    points = sample['inputs']['points']
    points = points[0] if isinstance(points, list) else points
    return points if isinstance(points, torch.Tensor) else points.tensor


def image_metrics(imgs: torch.Tensor) -> dict[str, float]:
    arr = imgs.detach().cpu().float().numpy()
    view_stats = []
    for view in arr:
        hwc = np.transpose(view, (1, 2, 0))
        gray = cv2.cvtColor(np.clip(hwc, 0, 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
        prob = hist / max(float(hist.sum()), 1.0)
        entropy = float(-(prob[prob > 0] * np.log2(prob[prob > 0])).sum())
        view_stats.append({
            'brightness': float(hwc.mean() / 255.0),
            'contrast': float(hwc.std() / 255.0),
            'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            'entropy': entropy,
        })
    return {key: mean(v[key] for v in view_stats) for key in view_stats[0]}


def lidar_metrics(points: torch.Tensor) -> dict[str, float]:
    pts = points.detach().cpu().float()
    xy = pts[:, :2]
    point_count = int(pts.shape[0])
    x_min, y_min, x_max, y_max, cell = -54.0, -54.0, 54.0, 54.0, 0.5
    valid = ((xy[:, 0] >= x_min) & (xy[:, 0] < x_max) & (xy[:, 1] >= y_min)
             & (xy[:, 1] < y_max))
    xy_valid = xy[valid]
    grid_w = int((x_max - x_min) / cell)
    grid_h = int((y_max - y_min) / cell)
    if len(xy_valid) > 0:
        ix = ((xy_valid[:, 0] - x_min) / cell).long().clamp(0, grid_w - 1)
        iy = ((xy_valid[:, 1] - y_min) / cell).long().clamp(0, grid_h - 1)
        linear = iy * grid_w + ix
        occupancy = int(torch.unique(linear).numel())
    else:
        occupancy = 0
    occupancy_ratio = occupancy / float(grid_w * grid_h)
    intensity_mean = float(pts[:, 3].mean()) if pts.shape[1] > 3 else 0.0
    intensity_std = float(pts[:, 3].std()) if pts.shape[1] > 3 else 0.0
    return {
        'point_count': point_count,
        'occupancy_ratio': occupancy_ratio,
        'intensity_mean': intensity_mean,
        'intensity_std': intensity_std,
    }


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def balance_score(value: float, reference: float) -> float:
    if value <= 0 or reference <= 0:
        return 0.0
    return clamp01(min(value / reference, reference / value))


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def score_rows(rows: list[dict], clean_rows: list[dict]) -> None:
    clean_ref = {
        'brightness': median([r['brightness'] for r in clean_rows]),
        'contrast': median([r['contrast'] for r in clean_rows]),
        'sharpness': median([r['sharpness'] for r in clean_rows]),
        'entropy': median([r['entropy'] for r in clean_rows]),
        'point_count': median([r['point_count'] for r in clean_rows]),
        'occupancy_ratio': median([r['occupancy_ratio'] for r in clean_rows]),
        'intensity_mean': median([r['intensity_mean'] for r in clean_rows]),
        'intensity_std': median([r['intensity_std'] for r in clean_rows]),
    }
    for row in rows:
        brightness_score = clamp01(
            1.0 - abs(row['brightness'] - clean_ref['brightness']) / 0.45)
        contrast_score = balance_score(row['contrast'], clean_ref['contrast'])
        sharpness_score = balance_score(row['sharpness'],
                                        clean_ref['sharpness'])
        entropy_score = balance_score(row['entropy'], clean_ref['entropy'])
        camera_reliability = (
            0.20 * brightness_score + 0.35 * contrast_score +
            0.35 * sharpness_score + 0.10 * entropy_score)

        density_score = balance_score(row['point_count'],
                                      clean_ref['point_count'])
        occupancy_score = balance_score(row['occupancy_ratio'],
                                        clean_ref['occupancy_ratio'])
        intensity_mean_score = balance_score(row['intensity_mean'],
                                             clean_ref['intensity_mean'])
        intensity_std_score = balance_score(row['intensity_std'],
                                            clean_ref['intensity_std'])
        lidar_reliability = (
            0.45 * density_score + 0.35 * occupancy_score +
            0.10 * intensity_mean_score + 0.10 * intensity_std_score)

        row['brightness_score'] = brightness_score
        row['contrast_score'] = contrast_score
        row['sharpness_score'] = sharpness_score
        row['entropy_score'] = entropy_score
        row['camera_reliability'] = camera_reliability
        row['density_score'] = density_score
        row['occupancy_score'] = occupancy_score
        row['intensity_mean_score'] = intensity_mean_score
        row['intensity_std_score'] = intensity_std_score
        row['lidar_reliability'] = lidar_reliability
        row['degradation_score_avg'] = 1.0 - (
            0.5 * camera_reliability + 0.5 * lidar_reliability)
        row['degradation_score_any'] = 1.0 - min(camera_reliability,
                                                 lidar_reliability)
        row['degradation_score'] = row['degradation_score_avg']


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row['exp_name'], row['corruption_type'],
               row['corruption_severity'])
        groups.setdefault(key, []).append(row)
    agg_rows = []
    for (exp_name, ctype, severity), items in groups.items():
        metric_keys = [
            'brightness', 'contrast', 'sharpness', 'entropy', 'point_count',
            'occupancy_ratio', 'camera_reliability', 'lidar_reliability',
            'degradation_score_avg', 'degradation_score_any',
            'degradation_score'
        ]
        agg = {
            'exp_name': exp_name,
            'corruption_type': ctype,
            'corruption_severity': severity,
            'sample_count': len(items),
        }
        for key in metric_keys:
            agg[f'mean_{key}'] = mean(float(item[key]) for item in items)
        agg_rows.append(agg)
    return agg_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_corruption_runtime_nus-3d.py')
    parser.add_argument('--repo-root', default='/workspace/mmdetection3d')
    parser.add_argument('--max-samples', type=int, default=81)
    parser.add_argument('--out-dir', default='/workspace/analysis_results')
    args = parser.parse_args()

    MMLogger.get_instance('mmengine').setLevel('ERROR')
    config_path = Path(args.config)
    repo_root = Path(args.repo_root)
    out_dir = Path(args.out_dir)

    rows = []
    for exp_name, ctype, severity, cam_corr, lidar_corr in DEFAULT_EXPERIMENTS:
        dataset = build_dataset(config_path, repo_root, exp_name, cam_corr,
                                lidar_corr, severity if severity != 'none' else 'mild')
        num_samples = min(args.max_samples, len(dataset))
        for idx in range(num_samples):
            sample = dataset[idx]
            img_stats = image_metrics(get_img_tensor(sample))
            lidar_stats = lidar_metrics(get_points_tensor(sample))
            rows.append({
                'exp_name': exp_name,
                'corruption_type': ctype,
                'corruption_severity': severity,
                'sample_index': idx,
                **img_stats,
                **lidar_stats,
            })

    clean_rows = [row for row in rows if row['exp_name'] == 'bevfusion_mini_clean_baseline']
    score_rows(rows, clean_rows)
    agg_rows = aggregate(rows)

    sample_csv = out_dir / 'reliability_proxy_samples.csv'
    sample_json = out_dir / 'reliability_proxy_samples.json'
    agg_csv = out_dir / 'reliability_proxy_summary.csv'
    agg_json = out_dir / 'reliability_proxy_summary.json'
    stage_json = out_dir / 'stage6_reliability_proxy_summary.json'

    write_csv(sample_csv, rows)
    write_csv(agg_csv, agg_rows)
    sample_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                           encoding='utf-8')
    agg_json.write_text(json.dumps(agg_rows, indent=2, ensure_ascii=False),
                        encoding='utf-8')

    stage = {
        'stage': 'Stage 6 - Reliability proxy analysis',
        'status': 'completed',
        'sample_csv': str(sample_csv),
        'sample_json': str(sample_json),
        'summary_csv': str(agg_csv),
        'summary_json': str(agg_json),
        'num_experiments': len(DEFAULT_EXPERIMENTS),
        'num_sample_rows': len(rows),
        'notes': [
            'Scores are normalized using the clean mini split median statistics.',
            'camera_reliability uses brightness, contrast, sharpness, and entropy.',
            'lidar_reliability uses point density and BEV occupancy.',
            'degradation_score is the average-reliability gate: 1 - 0.5 * (camera_reliability + lidar_reliability).',
            'degradation_score_any is also saved as 1 - min(camera_reliability, lidar_reliability) for any-modality-failure gating.'
        ],
    }
    stage_json.write_text(json.dumps(stage, indent=2, ensure_ascii=False),
                          encoding='utf-8')
    print(json.dumps(stage, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
