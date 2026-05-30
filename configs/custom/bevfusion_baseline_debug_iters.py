import os

_base_ = ['./bevfusion_baseline_1xb1_nuscenes.py']

# Short iter-based config for launch sanity checks. The full training config
# remains epoch-based; this file only verifies that data/model/optimizer run.
debug_max_iters = int(os.getenv('BEVFUSION_DEBUG_MAX_ITERS', '1'))
work_root = _base_.work_root
exp_name = os.getenv(
    'BEVFUSION_EXP_NAME',
    f'baseline_bevfusion_debug_{debug_max_iters}iter_20260522')
work_dir = os.path.join(work_root, exp_name)

train_cfg = dict(
    _delete_=True,
    by_epoch=False, max_iters=debug_max_iters, val_interval=debug_max_iters + 1)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0,
        by_epoch=False,
        begin=0,
        end=max(debug_max_iters, 1))
]

default_hooks = _base_.default_hooks
default_hooks['logger']['interval'] = int(
    os.getenv('BEVFUSION_LOG_INTERVAL', '1'))
default_hooks['checkpoint'] = dict(
    type='CheckpointHook',
    by_epoch=False,
    interval=debug_max_iters + 1,
    save_last=False,
    max_keep_ckpts=1)

train_dataloader = _base_.train_dataloader
train_dataloader['num_workers'] = int(os.getenv('BEVFUSION_NUM_WORKERS', '2'))
train_dataloader['persistent_workers'] = False

val_cfg = None
val_dataloader = None
val_evaluator = None
