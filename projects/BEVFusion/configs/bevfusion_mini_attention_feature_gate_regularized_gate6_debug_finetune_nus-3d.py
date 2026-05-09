_base_ = [
    './bevfusion_mini_attention_feature_gate_regularized_debug_finetune_nus-3d.py'
]

model = dict(
    attention_fusion_layer=dict(
        gate_init=-6.0,
        residual_loss_weight=0.05,
        gate_loss_weight=0.01))

work_dir = '/workspace/work_dirs/bevfusion_mini_attention_feature_gate_regularized_gate6_debug_finetune'
val_evaluator = dict(
    jsonfile_prefix=
    '/workspace/outputs/bevfusion_mini_attention_feature_gate_regularized_gate6_debug_finetune_val/results')
