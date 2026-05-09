#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path


PAIRS = [
    ("clean", "bevfusion_mini_clean_baseline",
     "bevfusion_mini_finetuned_iter10_clean"),
    ("camera_fog_severe", "bevfusion_mini_camera_fog_severe",
     "bevfusion_mini_finetuned_iter10_camera_fog_severe"),
    ("lidar_dropout_severe", "bevfusion_mini_lidar_dropout_severe",
     "bevfusion_mini_finetuned_iter10_lidar_dropout_severe"),
    ("camera_lidar_all_severe", "bevfusion_mini_camera_lidar_all_severe",
     "bevfusion_mini_finetuned_iter10_camera_lidar_all_severe"),
]


def main() -> None:
    rows = json.loads(
        Path("/workspace/analysis_results/experiment_summary.json").read_text(
            encoding="utf-8"))
    by_name = {row["exp_name"]: row for row in rows}
    out = []
    for condition, baseline_name, finetuned_name in PAIRS:
        baseline = by_name[baseline_name]
        finetuned = by_name[finetuned_name]
        out.append({
            "condition": condition,
            "baseline_exp_name": baseline_name,
            "finetuned_exp_name": finetuned_name,
            "baseline_mAP": baseline["mAP"],
            "finetuned_mAP": finetuned["mAP"],
            "delta_mAP": finetuned["mAP"] - baseline["mAP"],
            "baseline_NDS": baseline["NDS"],
            "finetuned_NDS": finetuned["NDS"],
            "delta_NDS": finetuned["NDS"] - baseline["NDS"],
            "baseline_detection_count": baseline["detection_count"],
            "finetuned_detection_count": finetuned["detection_count"],
            "baseline_mean_confidence": baseline["mean_confidence"],
            "finetuned_mean_confidence": finetuned["mean_confidence"],
        })

    csv_path = Path("/workspace/analysis_results/stage12_finetuned_iter10_eval.csv")
    json_path = Path("/workspace/analysis_results/stage12_finetuned_iter10_eval.json")
    summary_path = Path(
        "/workspace/analysis_results/stage12_finetuned_iter10_summary.json")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    summary = {
        "stage": "Stage 12",
        "status": "explicit_iter10_eval_completed",
        "checkpoint": "/workspace/work_dirs/bevfusion_mini_attention_feature_gate_debug_finetune/iter_10.pth",
        "experiments": out,
        "interpretation": [
            "This is a 10-iteration debug finetune, not a final robustness result.",
            "Clean degradation is the main safety signal to inspect before longer training.",
            "If clean degradation is large, reduce LR/gate strength or add clean consistency before scaling iterations."
        ]
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
