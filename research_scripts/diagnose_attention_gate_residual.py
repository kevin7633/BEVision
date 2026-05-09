#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean

import torch
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--camera-corruption', default='none')
    parser.add_argument('--lidar-corruption', default='none')
    parser.add_argument('--severity', default='clean')
    parser.add_argument('--max-samples', type=int, default=20)
    parser.add_argument(
        '--out-json',
        default='/workspace/analysis_results/attention_gate_residual_diagnostics.json')
    parser.add_argument(
        '--out-csv',
        default='/workspace/analysis_results/attention_gate_residual_diagnostics.csv')
    return parser.parse_args()


def unwrap_model(model):
    return getattr(model, 'module', model)


def attention_layer(model):
    model = unwrap_model(model)
    return getattr(model, 'attention_fusion_layer', None)


def main() -> None:
    args = parse_args()
    os.environ['BEVFUSION_EXP_NAME'] = args.exp_name
    os.environ['BEVFUSION_CAMERA_CORRUPTION'] = args.camera_corruption
    os.environ['BEVFUSION_LIDAR_CORRUPTION'] = args.lidar_corruption
    os.environ['BEVFUSION_CORRUPTION_SEVERITY'] = (
        'mild' if args.severity == 'clean' else args.severity)
    os.environ['BEVFUSION_CORRUPTION_SEED'] = '2026'

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.launcher = 'none'
    cfg.work_dir = f'/workspace/work_dirs/{args.exp_name}_diagnostic'
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    if hasattr(cfg, 'val_dataloader'):
        cfg.val_dataloader = cfg.test_dataloader

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    runner.model.eval()

    stats = []
    processed = 0
    for data_batch in runner.test_dataloader:
        if processed >= args.max_samples:
            break
        with torch.no_grad():
            runner.model.test_step(data_batch)
        layer = attention_layer(runner.model)
        if layer is None or layer.last_debug_stats is None:
            raise RuntimeError('No attention debug stats were produced.')
        row = dict(layer.last_debug_stats)
        row.update({
            'exp_name': args.exp_name,
            'camera_corruption': args.camera_corruption,
            'lidar_corruption': args.lidar_corruption,
            'severity': args.severity,
            'sample_index': processed,
        })
        stats.append(row)
        processed += 1

    numeric_keys = [
        'gate_mean', 'gate_min', 'gate_max', 'correction_abs_mean',
        'residual_abs_mean', 'base_abs_mean', 'residual_to_base_abs',
        'residual_to_base_l2'
    ]
    for optional_key in ('camera_reliability_mean', 'lidar_reliability_mean'):
        if stats and optional_key in stats[0]:
            numeric_keys.append(optional_key)
    summary = {
        'exp_name': args.exp_name,
        'checkpoint': args.checkpoint,
        'config': args.config,
        'camera_corruption': args.camera_corruption,
        'lidar_corruption': args.lidar_corruption,
        'severity': args.severity,
        'sample_count': len(stats),
        'means': {key: mean(row[key] for row in stats) for key in numeric_keys},
        'samples': stats,
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    existing = []
    if out_json.exists():
        existing = json.loads(out_json.read_text(encoding='utf-8'))
        if isinstance(existing, dict):
            existing = [existing]
    existing = [
        item for item in existing if item.get('exp_name') != args.exp_name
    ]
    existing.append(summary)
    out_json.write_text(json.dumps(existing, indent=2), encoding='utf-8')

    flat_rows = []
    for item in existing:
        row = {
            'exp_name': item['exp_name'],
            'checkpoint': item['checkpoint'],
            'config': item['config'],
            'camera_corruption': item['camera_corruption'],
            'lidar_corruption': item['lidar_corruption'],
            'severity': item['severity'],
            'sample_count': item['sample_count'],
        }
        row.update({f'mean_{k}': v for k, v in item['means'].items()})
        flat_rows.append(row)
    fieldnames = []
    for row in flat_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
