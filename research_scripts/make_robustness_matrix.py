#!/usr/bin/env python3
"""Create compact robustness matrix files from experiment_summary.json."""

from __future__ import annotations

import csv
import json
from pathlib import Path


SEVERITY_ORDER = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
TYPE_ORDER = {
    'none': 0,
    'camera_fog': 1,
    'lidar_random_dropout': 2,
    'camera_lidar_all': 3,
}


def main() -> None:
    src = Path('/workspace/analysis_results/experiment_summary.json')
    out_csv = Path('/workspace/analysis_results/stage5_robustness_matrix.csv')
    out_json = Path('/workspace/analysis_results/stage5_robustness_matrix.json')
    stage_json = Path('/workspace/analysis_results/stage5_baseline_robustness_summary.json')

    rows = json.load(src.open('r', encoding='utf-8'))
    clean = next(row for row in rows if row['exp_name'] == 'bevfusion_mini_clean_baseline')
    clean_map = float(clean['mAP'])
    clean_nds = float(clean['NDS'])

    selected = []
    for row in rows:
        ctype = row['corruption_type']
        severity = row['corruption_severity']
        if ctype not in TYPE_ORDER:
            continue
        item = {
            'exp_name': row['exp_name'],
            'corruption_type': ctype,
            'corruption_severity': severity,
            'mAP': float(row['mAP']),
            'NDS': float(row['NDS']),
            'delta_mAP': float(row['mAP']) - clean_map,
            'delta_NDS': float(row['NDS']) - clean_nds,
            'mean_confidence': float(row['mean_confidence']),
            'detection_count': int(row['detection_count']),
            'output_dir': row['output_dir'],
        }
        selected.append(item)

    selected.sort(key=lambda x: (
        TYPE_ORDER[x['corruption_type']],
        SEVERITY_ORDER[x['corruption_severity']],
        x['exp_name'],
    ))

    fields = [
        'exp_name', 'corruption_type', 'corruption_severity', 'mAP', 'NDS',
        'delta_mAP', 'delta_NDS', 'mean_confidence', 'detection_count',
        'output_dir'
    ]
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    with out_json.open('w', encoding='utf-8') as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    worst_by_nds = min(selected, key=lambda x: x['NDS'])
    summary = {
        'stage': 'Stage 5 - Baseline robustness experiment',
        'status': 'completed',
        'clean_reference': {
            'exp_name': clean['exp_name'],
            'mAP': clean_map,
            'NDS': clean_nds,
            'detection_count': int(clean['detection_count']),
        },
        'num_matrix_rows': len(selected),
        'matrix_csv': str(out_csv),
        'matrix_json': str(out_json),
        'worst_condition_by_nds': worst_by_nds,
        'notes': [
            'nuScenes-mini metrics are noisy because the validation split has 81 samples.',
            'camera_fog alone produced only small degradation in this setup.',
            'LiDAR dropout produced monotonic degradation as severity increased.',
            'camera_lidar_all is an intentionally strong stress test and caused severe collapse.',
        ],
    }
    with stage_json.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
