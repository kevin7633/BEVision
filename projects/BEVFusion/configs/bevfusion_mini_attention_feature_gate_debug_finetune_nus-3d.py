_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_attention_feature_gate_nus-3d.py'
]

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
backend_args = None
input_modality = dict(use_lidar=True, use_camera=True)
metainfo = dict(
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    version='v1.0-mini')

model = dict(
    train_attention_only=True,
    img_backbone=dict(init_cfg=None),
    attention_fusion_layer=dict(
        type='ReliabilityGuidedAttentionResidualFusion',
        in_channels=[80, 256],
        out_channels=256,
        use_attention_fusion=True,
        use_self_attention=True,
        use_reliability_gate=True,
        gate_type='feature',
        gate_hidden_dim=64,
        hidden_dim=128,
        num_heads=4,
        downsample_ratio=4,
        gate_init=-6.0,
        debug_shape=False))

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='BEVFusionCameraCorruption',
        corruption_type='all',
        severity='moderate',
        seed=2026,
        apply_prob=0.5),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='BEVFusionLiDARCorruption',
        corruption_type='all',
        severity='moderate',
        seed=2026,
        apply_prob=0.5,
        point_cloud_range=point_cloud_range),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=True),
    dict(type='BEVFusionGlobalRotScaleTrans',
         scale_ratio_range=[1.0, 1.0],
         rot_range=[0.0, 0.0],
         translation_std=0.0),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow',
            'pcd_rotation', 'pcd_scale_factor', 'pcd_trans',
            'img_aug_matrix', 'lidar_aug_matrix', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        dataset=dict(
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality)))

train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop', max_iters=10,
    val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = []
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10,
                    max_keep_ckpts=2))
val_evaluator = dict(
    jsonfile_prefix=
    '/workspace/outputs/bevfusion_mini_attention_feature_gate_debug_finetune_val/results')

load_from = '/workspace/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'
work_dir = '/workspace/work_dirs/bevfusion_mini_attention_feature_gate_debug_finetune'
randomness = dict(seed=2026, deterministic=False)
