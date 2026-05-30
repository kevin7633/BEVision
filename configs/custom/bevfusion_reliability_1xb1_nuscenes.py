import os

_base_ = ['./bevfusion_baseline_1xb1_nuscenes.py']

model = _base_.model
point_cloud_range = _base_.point_cloud_range
train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline
train_dataloader = _base_.train_dataloader
val_dataloader = _base_.val_dataloader
test_dataloader = _base_.test_dataloader
work_root = _base_.work_root
exp_name = _base_.exp_name

model.update(
    reliability_fusion_layer=dict(
        type='ReliabilityGuidedResidualFusion',
        in_channels=[80, 256],
        out_channels=256,
        enabled=True,
        base_source='lidar',
        correction_source='camera_lidar',
        hidden_channels=256,
        use_reliability_gate=True,
        gate_mode='image_lidar',
        gate_formula='sigmoid',
        alpha=4.0,
        beta=2.0,
        bias=-1.0,
        gate_min=0.0,
        gate_max=0.75,
        use_channel_attention=False,
        use_spatial_attention=False,
        residual_loss_weight=0.0,
        gate_loss_weight=0.0))

reliability_meta_keys = [
    'image_reliability', 'image_reliability_per_view',
    'image_brightness_per_view', 'image_contrast_per_view',
    'image_sharpness_per_view', 'lidar_reliability', 'lidar_point_count',
    'lidar_reliability_map', 'camera_corruption', 'lidar_corruption'
]
reliability_transform = dict(
    type='BEVFusionSensorReliability',
    lidar_count_ref=300000.0,
    compute_lidar_bev_map=False,
    point_cloud_range=point_cloud_range)


def _insert_reliability(pipeline):
    out = []
    inserted = False
    for step in pipeline:
        step = dict(step)
        if step['type'] == 'Pack3DDetInputs':
            meta_keys = list(step.get('meta_keys', []))
            for key in reliability_meta_keys:
                if key not in meta_keys:
                    meta_keys.append(key)
            step['meta_keys'] = meta_keys
            out.append(reliability_transform)
            inserted = True
        out.append(step)
    if not inserted:
        out.append(reliability_transform)
    return out


train_pipeline = _insert_reliability(train_pipeline)
test_pipeline = _insert_reliability(test_pipeline)
train_dataloader['dataset']['dataset']['pipeline'] = train_pipeline
val_dataloader['dataset']['pipeline'] = test_pipeline
test_dataloader = val_dataloader

exp_name = exp_name.replace('baseline', 'reliability')
work_dir = os.path.join(work_root, exp_name)
