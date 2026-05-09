#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/workspace/mmdetection3d
PYTHON=/workspace/envs/bevfusion/bin/python
CONFIG=${1:-/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_attention_nus-3d.py}
CHECKPOINT=${2:-/workspace/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth}
SAMPLE_INDEX=${3:-0}

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

"${PYTHON}" /workspace/scripts/smoke_attention_fusion_module.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --sample-index "${SAMPLE_INDEX}" \
  --device cuda:0
