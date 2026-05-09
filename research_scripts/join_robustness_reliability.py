#!/usr/bin/env python3
"""Join Stage 5 performance and Stage 6 reliability summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    robust_path = Path('/workspace/analysis_results/stage5_robustness_matrix.csv')
    rel_path = Path('/workspace/analysis_results/reliability_proxy_summary.csv')
    out_csv = Path('/workspace/analysis_results/stage6_reliability_vs_performance.csv')
    out_json = Path('/workspace/analysis_results/stage6_reliability_vs_performance.json')

    robust_rows = {row['exp_name']: row for row in read_csv(robust_path)}
    rel_rows = read_csv(rel_path)
    joined = []
    for rel in rel_rows:
        perf = robust_rows.get(rel['exp_name'])
        if perf is None:
            continue
        joined.append({
            'exp_name': rel['exp_name'],
            'corruption_type': rel['corruption_type'],
            'corruption_severity': rel['corruption_severity'],
            'mAP': perf['mAP'],
            'NDS': perf['NDS'],
            'delta_mAP': perf['delta_mAP'],
            'delta_NDS': perf['delta_NDS'],
            'mean_camera_reliability': rel['mean_camera_reliability'],
            'mean_lidar_reliability': rel['mean_lidar_reliability'],
            'mean_degradation_score_avg': rel['mean_degradation_score_avg'],
            'mean_degradation_score_any': rel['mean_degradation_score_any'],
            'mean_point_count': rel['mean_point_count'],
            'mean_occupancy_ratio': rel['mean_occupancy_ratio'],
        })

    write_csv(out_csv, joined)
    out_json.write_text(json.dumps(joined, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(json.dumps({
        'joined_csv': str(out_csv),
        'joined_json': str(out_json),
        'rows': len(joined),
    }, indent=2))


if __name__ == '__main__':
    main()
