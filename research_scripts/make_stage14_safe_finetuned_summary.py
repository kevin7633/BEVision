#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path


PAIRS = [
    ("clean", "bevfusion_mini_clean_baseline",
     "bevfusion_mini_safe_finetuned_iter10_clean"),
    ("camera_fog_severe", "bevfusion_mini_camera_fog_severe",
     "bevfusion_mini_safe_finetuned_iter10_camera_fog_severe"),
    ("lidar_dropout_severe", "bevfusion_mini_lidar_dropout_severe",
     "bevfusion_mini_safe_finetuned_iter10_lidar_dropout_severe"),
    ("camera_lidar_all_severe", "bevfusion_mini_camera_lidar_all_severe",
     "bevfusion_mini_safe_finetuned_iter10_camera_lidar_all_severe"),
]


def main() -> None:
    rows = json.loads(
        Path("/workspace/analysis_results/experiment_summary.json").read_text(
            encoding="utf-8"))
    by_name = {row["exp_name"]: row for row in rows}
    out = []
    for condition, baseline_name, safe_name in PAIRS:
        baseline = by_name[baseline_name]
        safe = by_name[safe_name]
        out.append({
            "condition": condition,
            "baseline_exp_name": baseline_name,
            "safe_finetuned_exp_name": safe_name,
            "baseline_mAP": baseline["mAP"],
            "safe_finetuned_mAP": safe["mAP"],
            "delta_mAP": safe["mAP"] - baseline["mAP"],
            "baseline_NDS": baseline["NDS"],
            "safe_finetuned_NDS": safe["NDS"],
            "delta_NDS": safe["NDS"] - baseline["NDS"],
            "baseline_detection_count": baseline["detection_count"],
            "safe_finetuned_detection_count": safe["detection_count"],
            "baseline_mean_confidence": baseline["mean_confidence"],
            "safe_finetuned_mean_confidence": safe["mean_confidence"],
        })

    csv_path = Path("/workspace/analysis_results/stage14_safe_finetuned_eval.csv")
    json_path = Path("/workspace/analysis_results/stage14_safe_finetuned_eval.json")
    summary_path = Path(
        "/workspace/analysis_results/stage14_safe_finetuned_summary.json")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    summary = {
        "stage": "Stage 14",
        "status": "safe_iter10_corruption_eval_completed",
        "checkpoint": "/workspace/work_dirs/bevfusion_mini_attention_feature_gate_safe_debug_finetune/iter_10.pth",
        "experiments": out,
        "interpretation": [
            "This is still a 10-iteration debug result on nuScenes-mini.",
            "Use the clean delta as the primary safety constraint.",
            "A useful setting should keep clean degradation small while improving at least one corrupted condition consistently."
        ]
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
