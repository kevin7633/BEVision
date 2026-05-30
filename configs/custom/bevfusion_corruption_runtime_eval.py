import os

_base_ = ['./bevfusion_reliability_1xb1_nuscenes.py']

test_pipeline = _base_.test_pipeline
val_dataloader = _base_.val_dataloader
test_dataloader = _base_.test_dataloader
work_root = _base_.work_root

camera_corruption = os.getenv('BEVFUSION_CAMERA_CORRUPTION', 'none')
lidar_corruption = os.getenv('BEVFUSION_LIDAR_CORRUPTION', 'none')
severity = os.getenv('BEVFUSION_CORRUPTION_SEVERITY', 'moderate')
seed = int(os.getenv('BEVFUSION_CORRUPTION_SEED', '2026'))


def _insert_before_reliability(pipeline, transform):
    out = []
    for step in pipeline:
        if step['type'] == 'BEVFusionSensorReliability':
            out.append(transform)
        out.append(step)
    return out


if camera_corruption != 'none':
    camera_transform = dict(
        type='BEVFusionCameraCorruption',
        corruption_type=camera_corruption,
        severity=severity,
        seed=seed)
    test_pipeline = _insert_before_reliability(test_pipeline,
                                               camera_transform)

if lidar_corruption != 'none':
    lidar_transform = dict(
        type='BEVFusionLiDARCorruption',
        corruption_type=lidar_corruption,
        severity=severity,
        seed=seed)
    test_pipeline = _insert_before_reliability(test_pipeline, lidar_transform)

val_dataloader['dataset']['pipeline'] = test_pipeline
test_dataloader = val_dataloader
exp_name = os.getenv(
    'BEVFUSION_EXP_NAME',
    f'corruption_eval_cam-{camera_corruption}_lidar-{lidar_corruption}_{severity}'
)
work_dir = os.path.join(work_root, exp_name)
