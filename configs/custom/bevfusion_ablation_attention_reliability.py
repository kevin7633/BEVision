import os

_base_ = ['./bevfusion_reliability_1xb1_nuscenes.py']

model = _base_.model
work_root = _base_.work_root

model['reliability_fusion_layer'].update(
    use_channel_attention=True,
    use_spatial_attention=True)
exp_name = 'ablation_attention_reliability_20260521'
work_dir = os.path.join(work_root, exp_name)
