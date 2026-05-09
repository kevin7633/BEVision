#!/usr/bin/env bash
set -euo pipefail

source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion

cd /workspace/mmdetection3d

export PYTHONPATH=/workspace/mmdetection3d:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG=/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_attention_feature_gate_safe_debug_finetune_nus-3d.py
LOG=/workspace/outputs/bevfusion_mini_attention_feature_gate_safe_debug_finetune.log

python tools/train.py "$CONFIG" 2>&1 | tee "$LOG"
