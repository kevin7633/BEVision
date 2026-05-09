_base_ = [
    './bevfusion_mini_attention_feature_gate_debug_finetune_nus-3d.py'
]

model = dict(
    attention_fusion_layer=dict(
        gate_init=-8.0))

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=None),
    dict(
        type='BEVFusionCameraCorruption',
        corruption_type='all',
        severity='moderate',
        seed=2026,
        apply_prob=0.25),
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
        type='BEVFusionLiDARCorruption',
        corruption_type='all',
        severity='moderate',
        seed=2026,
        apply_prob=0.25,
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
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
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[1.0, 1.0],
        rot_range=[0.0, 0.0],
        translation_std=0.0),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
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
    dataset=dict(dataset=dict(pipeline=train_pipeline)))

train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop', max_iters=10,
    val_interval=100)
optim_wrapper = dict(
    optimizer=dict(lr=1e-5))
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2))
val_evaluator = dict(
    jsonfile_prefix=
    '/workspace/outputs/bevfusion_mini_attention_feature_gate_safe_debug_finetune_val/results')

work_dir = '/workspace/work_dirs/bevfusion_mini_attention_feature_gate_safe_debug_finetune'
