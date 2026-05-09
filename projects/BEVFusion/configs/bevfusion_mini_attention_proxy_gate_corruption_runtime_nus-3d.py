_base_ = [
    './bevfusion_mini_attention_feature_gate_corruption_runtime_nus-3d.py'
]

model = dict(
    attention_fusion_layer=dict(
        gate_type='proxy',
        proxy_gate_min=0.0,
        proxy_gate_scale=0.20,
        proxy_gate_power=1.0,
        gate_init=-6.0))

work_dir = '/workspace/work_dirs/bevfusion_mini_attention_proxy_gate_corruption_runtime'
