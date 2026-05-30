import os

_base_ = [
    '../../projects/BEVFusion/configs/'
    'bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

data_root = os.getenv(
    'NUSCENES_DATA_ROOT',
    'data/nuscenes/')
work_root = os.getenv(
    'BEVFUSION_WORK_ROOT',
    'work_dirs')
exp_name = os.getenv('BEVFUSION_EXP_NAME',
                     'baseline_bevfusion_full_trainval_20260521')
work_dir = os.path.join(work_root, exp_name)

point_cloud_range = _base_.point_cloud_range
train_dataloader = _base_.train_dataloader
val_dataloader = _base_.val_dataloader
test_dataloader = _base_.test_dataloader
val_evaluator = _base_.val_evaluator
test_evaluator = _base_.test_evaluator
default_hooks = _base_.default_hooks

train_dataloader['batch_size'] = int(os.getenv('BEVFUSION_BATCH_SIZE', '1'))
train_dataloader['num_workers'] = int(os.getenv('BEVFUSION_NUM_WORKERS', '2'))
train_dataloader['persistent_workers'] = False
train_dataloader['dataset']['dataset']['data_root'] = data_root
train_dataloader['dataset']['dataset']['ann_file'] = 'nuscenes_infos_train.pkl'

val_dataloader['batch_size'] = 1
val_dataloader['num_workers'] = int(os.getenv('BEVFUSION_NUM_WORKERS', '2'))
val_dataloader['persistent_workers'] = False
val_dataloader['dataset']['data_root'] = data_root
val_dataloader['dataset']['ann_file'] = 'nuscenes_infos_val.pkl'
test_dataloader = val_dataloader

val_evaluator['data_root'] = data_root
val_evaluator['ann_file'] = data_root + 'nuscenes_infos_val.pkl'
test_evaluator = val_evaluator

default_hooks['logger']['interval'] = int(os.getenv('BEVFUSION_LOG_INTERVAL',
                                                   '20'))
default_hooks['checkpoint'] = dict(
    type='CheckpointHook', interval=1, max_keep_ckpts=3)
