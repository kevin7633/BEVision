#!/usr/bin/env bash
set -euo pipefail

source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion

cd /workspace/mmdetection3d

export PYTHONPATH=/workspace/mmdetection3d:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG=/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_attention_feature_gate_regularized_clean_eval_nus-3d.py
CHECKPOINT=${1:-/workspace/work_dirs/bevfusion_mini_attention_feature_gate_regularized_debug_finetune/iter_10.pth}
LOG=/workspace/outputs/bevfusion_mini_regularized_finetuned_iter10_clean.log

python tools/test.py "$CONFIG" "$CHECKPOINT" 2>&1 | tee "$LOG"

python /workspace/scripts/summarize_bevfusion_results.py \
  --exp-name bevfusion_mini_regularized_finetuned_iter10_clean \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --dataset-version v1.0-mini \
  --corruption-type none \
  --corruption-severity clean \
  --clean-or-corrupted clean \
  --use-attention-fusion \
  --use-self-attention \
  --use-reliability-gate \
  --notes "Stage 15 regularized low-lr gate-init -8 attention-only finetune clean preservation check." \
  --output-dir /workspace/outputs/bevfusion_mini_regularized_finetuned_iter10_clean/results/pred_instances_3d \
  --metrics /workspace/outputs/bevfusion_mini_regularized_finetuned_iter10_clean/results/pred_instances_3d/metrics_summary.json \
  --predictions /workspace/outputs/bevfusion_mini_regularized_finetuned_iter10_clean/results/pred_instances_3d/results_nusc.json
