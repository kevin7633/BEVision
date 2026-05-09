#!/usr/bin/env bash
set -euo pipefail

source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion

export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/checkpoints/torch_cache

CONFIG="/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_attention_feature_gate_debug_finetune_nus-3d.py"
WORK_DIR="/workspace/work_dirs/bevfusion_mini_attention_feature_gate_debug_finetune"
LOG_FILE="/workspace/outputs/bevfusion_mini_attention_feature_gate_debug_finetune.log"

mkdir -p "${WORK_DIR}" /workspace/outputs

cd /workspace/mmdetection3d
python tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --launcher none \
  2>&1 | tee "${LOG_FILE}"
