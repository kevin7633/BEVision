import os

_base_ = ['./bevfusion_reliability_1xb1_nuscenes.py']

model = _base_.model
work_root = _base_.work_root

model['reliability_fusion_layer'].update(
    gate_mode='lidar_only',
    gate_formula='sigmoid',
    alpha=4.0,
    beta=0.0,
    bias=-2.0)
exp_name = 'ablation_lidar_gate_20260521'
work_dir = os.path.join(work_root, exp_name)
