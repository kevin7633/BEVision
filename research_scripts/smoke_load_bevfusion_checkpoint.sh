#!/usr/bin/env bash
set -euo pipefail

source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion

cd /workspace/mmdetection3d

python - <<'PY'
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS

config_path = 'projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
checkpoint_path = '/workspace/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'

init_default_scope('mmdet3d')
cfg = Config.fromfile(config_path)
if 'img_backbone' in cfg.model:
    cfg.model.img_backbone.init_cfg = None
model = MODELS.build(cfg.model)
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu', strict=False)

meta = checkpoint.get('meta', {}) if isinstance(checkpoint, dict) else {}
print('config:', config_path)
print('checkpoint:', checkpoint_path)
print('checkpoint_meta_keys:', sorted(meta.keys()))
print('dataset_meta_classes:', meta.get('dataset_meta', {}).get('classes', 'N/A'))
print('model_type:', type(model).__name__)
print('smoke_load: OK')
PY
