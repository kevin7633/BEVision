#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/checkpoints

url="https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
out="/workspace/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"

if [ -s "${out}" ]; then
  echo "Checkpoint already exists: ${out}"
else
  wget -c "${url}" -O "${out}"
fi

ls -lh "${out}"
