_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_attention_feature_gate_nus-3d.py'
]

model = dict(
    img_backbone=dict(init_cfg=None),
    attention_fusion_layer=dict(gate_init=-8.0))

test_evaluator = dict(
    jsonfile_prefix=
    '/workspace/outputs/bevfusion_mini_safe_finetuned_iter10_clean/results')

work_dir = '/workspace/work_dirs/bevfusion_mini_safe_finetuned_iter10_clean'

randomness = dict(seed=2026, deterministic=False)
