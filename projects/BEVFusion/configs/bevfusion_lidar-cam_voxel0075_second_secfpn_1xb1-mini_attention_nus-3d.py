_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_nus-3d.py'
]

model = dict(
    img_backbone=dict(init_cfg=None),
    attention_fusion_layer=dict(
        type='ReliabilityGuidedAttentionResidualFusion',
        in_channels=[80, 256],
        out_channels=256,
        use_attention_fusion=True,
        use_self_attention=True,
        use_reliability_gate=True,
        hidden_dim=128,
        num_heads=4,
        downsample_ratio=4,
        gate_init=-6.0,
        debug_shape=False))

test_evaluator = dict(
    jsonfile_prefix='/workspace/outputs/bevfusion_mini_attention_clean/results')

work_dir = '/workspace/work_dirs/bevfusion_mini_attention_clean'

randomness = dict(seed=2026, deterministic=False)
