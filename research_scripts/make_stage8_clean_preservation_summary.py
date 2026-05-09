#!/usr/bin/env python
"""Create Stage 8 clean preservation comparison from experiment summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BASELINE = "bevfusion_mini_clean_baseline"
EXPERIMENTS = [
    BASELINE,
    "bevfusion_mini_attention_off_clean",
    "bevfusion_mini_attention_clean",
]


def main() -> None:
    summary_path = Path("/workspace/analysis_results/experiment_summary.json")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    by_name = {row["exp_name"]: row for row in rows}
    baseline = by_name[BASELINE]

    out_rows = []
    for name in EXPERIMENTS:
        row = by_name[name]
        out_rows.append({
            "exp_name": name,
            "mAP": row["mAP"],
            "NDS": row["NDS"],
            "delta_mAP_vs_baseline": row["mAP"] - baseline["mAP"],
            "delta_NDS_vs_baseline": row["NDS"] - baseline["NDS"],
            "mean_confidence": row["mean_confidence"],
            "detection_count": row["detection_count"],
            "use_attention_fusion": row["use_attention_fusion"],
            "use_self_attention": row["use_self_attention"],
            "use_reliability_gate": row["use_reliability_gate"],
            "output_dir": row["output_dir"],
        })

    csv_path = Path("/workspace/analysis_results/stage8_clean_preservation.csv")
    json_path = Path("/workspace/analysis_results/stage8_clean_preservation.json")
    stage_path = Path("/workspace/analysis_results/stage8_clean_preservation_summary.json")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    json_path.write_text(
        json.dumps(out_rows, indent=2, ensure_ascii=False),
        encoding="utf-8")

    stage_summary = {
        "stage": "Stage 8",
        "status": "clean_preservation_checked",
        "baseline_exp_name": BASELINE,
        "experiments": out_rows,
        "interpretation": [
            "attention-off and attention-on both completed clean mini evaluation",
            "attention-on did not show clean performance collapse",
            "small differences should be interpreted cautiously because nuScenes-mini has only 81 validation samples and the current test script did not yet fix MMEngine runtime seed"
        ],
        "next_recommendation": [
            "fix randomness.seed in evaluation configs/scripts for repeatable comparisons",
            "run attention-on under selected corrupted settings before training to verify the zero-initialized residual behaves like baseline",
            "then add a reliability-conditioned gate or short fine-tuning experiment"
        ]
    }
    stage_path.write_text(
        json.dumps(stage_summary, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print(json.dumps(stage_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
