_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

model = dict(img_backbone=dict(init_cfg=None))

metainfo = dict(
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    version='v1.0-mini')

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(metainfo=metainfo))
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(metainfo=metainfo))

test_evaluator = dict(
    jsonfile_prefix='/workspace/outputs/bevfusion_mini_clean/results')

work_dir = '/workspace/work_dirs/bevfusion_mini_clean'

randomness = dict(seed=2026, deterministic=False)
