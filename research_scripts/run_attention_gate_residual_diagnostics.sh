#!/usr/bin/env bash
set -euo pipefail

source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion

export PYTHONPATH=/workspace/mmdetection3d:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG=/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_attention_feature_gate_corruption_runtime_nus-3d.py
CHECKPOINT=${1:-/workspace/work_dirs/bevfusion_mini_attention_feature_gate_regularized_debug_finetune/iter_10.pth}
MAX_SAMPLES=${MAX_SAMPLES:-20}

python /workspace/scripts/diagnose_attention_gate_residual.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --exp-name attention_diag_regularized_iter10_clean \
  --camera-corruption none \
  --lidar-corruption none \
  --severity clean \
  --max-samples "$MAX_SAMPLES"

python /workspace/scripts/diagnose_attention_gate_residual.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --exp-name attention_diag_regularized_iter10_camera_fog_severe \
  --camera-corruption fog \
  --lidar-corruption none \
  --severity severe \
  --max-samples "$MAX_SAMPLES"

python /workspace/scripts/diagnose_attention_gate_residual.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --exp-name attention_diag_regularized_iter10_lidar_dropout_severe \
  --camera-corruption none \
  --lidar-corruption random_dropout \
  --severity severe \
  --max-samples "$MAX_SAMPLES"

python /workspace/scripts/diagnose_attention_gate_residual.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --exp-name attention_diag_regularized_iter10_camera_lidar_all_severe \
  --camera-corruption all \
  --lidar-corruption all \
  --severity severe \
  --max-samples "$MAX_SAMPLES"
