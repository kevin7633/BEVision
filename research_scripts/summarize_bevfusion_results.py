#!/usr/bin/env python3
"""Summarize a BEVFusion nuScenes evaluation into analysis_results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


FIELDS = [
    "exp_name",
    "config",
    "checkpoint",
    "dataset_version",
    "corruption_type",
    "corruption_severity",
    "use_attention_fusion",
    "use_self_attention",
    "use_reliability_gate",
    "clean_or_corrupted",
    "mAP",
    "NDS",
    "mean_confidence",
    "confidence_std",
    "detection_count",
    "sample_count",
    "notes",
    "output_dir",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_existing_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "experiments" in data:
        return list(data["experiments"])
    raise ValueError(f"Unsupported summary JSON structure: {path}")


def upsert(rows: list[dict], new_row: dict) -> list[dict]:
    key = new_row["exp_name"]
    return [row for row in rows if row.get("exp_name") != key] + [new_row]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-version", default="v1.0-mini")
    parser.add_argument("--corruption-type", default="none")
    parser.add_argument("--corruption-severity", default="none")
    parser.add_argument("--clean-or-corrupted", default="clean")
    parser.add_argument("--use-attention-fusion", action="store_true")
    parser.add_argument("--use-self-attention", action="store_true")
    parser.add_argument("--use-reliability-gate", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--summary-csv", default="/workspace/analysis_results/experiment_summary.csv")
    parser.add_argument("--summary-json", default="/workspace/analysis_results/experiment_summary.json")
    parser.add_argument("--class-metrics-json", default="")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    predictions_path = Path(args.predictions)
    metrics = load_json(metrics_path)
    predictions = load_json(predictions_path)

    detections = [
        det
        for sample_dets in predictions.get("results", {}).values()
        for det in sample_dets
    ]
    scores = [float(det["detection_score"]) for det in detections if "detection_score" in det]

    row = {
        "exp_name": args.exp_name,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "dataset_version": args.dataset_version,
        "corruption_type": args.corruption_type,
        "corruption_severity": args.corruption_severity,
        "use_attention_fusion": args.use_attention_fusion,
        "use_self_attention": args.use_self_attention,
        "use_reliability_gate": args.use_reliability_gate,
        "clean_or_corrupted": args.clean_or_corrupted,
        "mAP": metrics.get("mean_ap"),
        "NDS": metrics.get("nd_score"),
        "mean_confidence": mean(scores) if scores else None,
        "confidence_std": pstdev(scores) if len(scores) > 1 else 0.0,
        "detection_count": len(detections),
        "sample_count": len(predictions.get("results", {})),
        "notes": args.notes,
        "output_dir": args.output_dir,
    }

    summary_csv = Path(args.summary_csv)
    summary_json = Path(args.summary_json)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if summary_csv.exists() and summary_csv.stat().st_size > 0:
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows = upsert(rows, row)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    json_rows = upsert(read_existing_json(summary_json), row)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2, ensure_ascii=False)

    if args.class_metrics_json:
        class_metrics_path = Path(args.class_metrics_json)
        class_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        class_summary = {
            "exp_name": args.exp_name,
            "mean_ap": metrics.get("mean_ap"),
            "nd_score": metrics.get("nd_score"),
            "label_aps": metrics.get("label_aps", {}),
            "mean_dist_aps": metrics.get("mean_dist_aps", {}),
            "tp_errors": metrics.get("tp_errors", {}),
            "label_tp_errors": metrics.get("label_tp_errors", {}),
        }
        with class_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(class_summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
