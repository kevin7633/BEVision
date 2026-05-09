#!/usr/bin/env python
"""Compare baseline and untrained attention under selected corruptions."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PAIRS = [
    ("camera_fog_severe", "bevfusion_mini_camera_fog_severe",
     "bevfusion_mini_attention_camera_fog_severe"),
    ("lidar_dropout_severe", "bevfusion_mini_lidar_dropout_severe",
     "bevfusion_mini_attention_lidar_dropout_severe"),
    ("camera_lidar_all_severe", "bevfusion_mini_camera_lidar_all_severe",
     "bevfusion_mini_attention_camera_lidar_all_severe"),
]


def main() -> None:
    rows = json.loads(
        Path("/workspace/analysis_results/experiment_summary.json").read_text(
            encoding="utf-8"))
    by_name = {row["exp_name"]: row for row in rows}
    out_rows = []
    for condition, baseline_name, attention_name in PAIRS:
        baseline = by_name[baseline_name]
        attention = by_name[attention_name]
        out_rows.append({
            "condition": condition,
            "baseline_exp_name": baseline_name,
            "attention_exp_name": attention_name,
            "baseline_mAP": baseline["mAP"],
            "attention_mAP": attention["mAP"],
            "delta_mAP_attention_minus_baseline":
            attention["mAP"] - baseline["mAP"],
            "baseline_NDS": baseline["NDS"],
            "attention_NDS": attention["NDS"],
            "delta_NDS_attention_minus_baseline":
            attention["NDS"] - baseline["NDS"],
            "baseline_detection_count": baseline["detection_count"],
            "attention_detection_count": attention["detection_count"],
            "baseline_mean_confidence": baseline["mean_confidence"],
            "attention_mean_confidence": attention["mean_confidence"],
        })

    csv_path = Path(
        "/workspace/analysis_results/stage9_prefinetune_attention_corruption.csv")
    json_path = Path(
        "/workspace/analysis_results/stage9_prefinetune_attention_corruption.json")
    summary_path = Path(
        "/workspace/analysis_results/stage9_prefinetune_attention_summary.json")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    json_path.write_text(
        json.dumps(out_rows, indent=2, ensure_ascii=False),
        encoding="utf-8")

    summary = {
        "stage": "Stage 9 pre-finetune check",
        "status": "selected_corruption_attention_eval_done",
        "experiments": out_rows,
        "interpretation": [
            "the current attention module is untrained and zero-initialized, so it is expected to behave close to baseline",
            "any true robustness gain should be evaluated only after reliability gating or finetuning is introduced",
            "nuScenes-mini metrics are sensitive, so small deltas should not be treated as improvement claims"
        ],
        "next_recommendation": [
            "connect reliability proxy to the residual gate or add a training-time scalar gate path",
            "run a short debug finetune on mini/subset with clean and corrupted samples",
            "compare clean preservation and severe corruption again after finetuning"
        ]
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
