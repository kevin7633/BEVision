_base_ = [
    './bevfusion_mini_attention_feature_gate_regularized_clean_eval_nus-3d.py'
]

model = dict(
    attention_fusion_layer=dict(
        gate_init=-6.0,
        residual_loss_weight=0.05,
        gate_loss_weight=0.01))

test_evaluator = dict(
    jsonfile_prefix=
    '/workspace/outputs/bevfusion_mini_regularized_gate6_finetuned_iter10_clean/results')

work_dir = '/workspace/work_dirs/bevfusion_mini_regularized_gate6_finetuned_iter10_clean'
