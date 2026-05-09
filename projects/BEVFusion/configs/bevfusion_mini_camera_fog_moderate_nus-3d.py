_base_ = ['./bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_nus-3d.py']

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
metainfo = dict(
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    version='v1.0-mini')

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=None),
    dict(
        type='BEVFusionCameraCorruption',
        corruption_type='fog',
        severity='moderate',
        seed=2026),
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
        backend_args=None),
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
    jsonfile_prefix='/workspace/outputs/bevfusion_mini_camera_fog_moderate/results')
work_dir = '/workspace/work_dirs/bevfusion_mini_camera_fog_moderate'
