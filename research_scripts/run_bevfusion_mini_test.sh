#!/usr/bin/env bash
set -euo pipefail

source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion

export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/checkpoints/torch_cache
mkdir -p "${TORCH_HOME}"

cd /workspace/mmdetection3d

config="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_nus-3d.py"
checkpoint="/workspace/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
work_dir="/workspace/work_dirs/bevfusion_mini_clean"
json_prefix="/workspace/outputs/bevfusion_mini_clean/results"

mkdir -p "${work_dir}" "$(dirname "${json_prefix}")"

python tools/test.py "${config}" "${checkpoint}" \
  --work-dir "${work_dir}" \
  --task multi-modality_det \
  --cfg-options \
    test_dataloader.batch_size=1 \
    test_dataloader.num_workers=0 \
    test_dataloader.persistent_workers=False \
    val_dataloader.batch_size=1 \
    val_dataloader.num_workers=0 \
    val_dataloader.persistent_workers=False \
    test_evaluator.jsonfile_prefix="${json_prefix}"
