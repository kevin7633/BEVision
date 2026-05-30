#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-data/nuscenes/}"
export BEVFUSION_WORK_ROOT="${BEVFUSION_WORK_ROOT:-work_dirs}"
export BEVFUSION_BATCH_SIZE="${BEVFUSION_BATCH_SIZE:-1}"
export BEVFUSION_NUM_WORKERS="${BEVFUSION_NUM_WORKERS:-2}"
export BEVFUSION_LOG_INTERVAL="${BEVFUSION_LOG_INTERVAL:-20}"

echo "[1/2] Training baseline BEVFusion"
export BEVFUSION_EXP_NAME="${BASELINE_EXP_NAME:-baseline_bevfusion_full_trainval}"
python tools/train.py configs/custom/bevfusion_baseline_1xb1_nuscenes.py

echo "[2/2] Training reliability-guided BEVFusion"
export BEVFUSION_EXP_NAME="${RELIABILITY_EXP_NAME:-reliability_bevfusion_full_trainval}"
python tools/train.py configs/custom/bevfusion_reliability_1xb1_nuscenes.py
