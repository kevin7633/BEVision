import os

_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_attention_nus-3d.py'
]

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
metainfo = dict(
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    version='v1.0-mini')

camera_corruption_type = os.environ.get('BEVFUSION_CAMERA_CORRUPTION', 'none')
lidar_corruption_type = os.environ.get('BEVFUSION_LIDAR_CORRUPTION', 'none')
corruption_severity = os.environ.get('BEVFUSION_CORRUPTION_SEVERITY', 'mild')
exp_name = os.environ.get(
    'BEVFUSION_EXP_NAME',
    f'bevfusion_mini_attention_cam-{camera_corruption_type}_lidar-'
    f'{lidar_corruption_type}_{corruption_severity}')
corruption_seed = int(os.environ.get('BEVFUSION_CORRUPTION_SEED', '2026'))

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=None)
]

if camera_corruption_type != 'none':
    test_pipeline.append(
        dict(
            type='BEVFusionCameraCorruption',
            corruption_type=camera_corruption_type,
            severity=corruption_severity,
            seed=corruption_seed))

test_pipeline += [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=None),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=None)
]

if lidar_corruption_type != 'none':
    test_pipeline.append(
        dict(
            type='BEVFusionLiDARCorruption',
            corruption_type=lidar_corruption_type,
            severity=corruption_severity,
            seed=corruption_seed,
            point_cloud_range=point_cloud_range))

test_pipeline += [
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
val_dataloader = test_dataloader

test_evaluator = dict(
    jsonfile_prefix=f'/workspace/outputs/{exp_name}/results')
work_dir = f'/workspace/work_dirs/{exp_name}'
randomness = dict(seed=2026, deterministic=False)
