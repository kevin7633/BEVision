#!/usr/bin/env bash
set -euo pipefail

source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion

export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/checkpoints/torch_cache

CONFIG="${1:-/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_camera_fog_moderate_nus-3d.py}"
CHECKPOINT="${2:-/workspace/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth}"

cd /workspace/mmdetection3d
python tools/test.py "${CONFIG}" "${CHECKPOINT}" --launcher none
