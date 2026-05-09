# BEVFusion Robustness Project Resume Notes

Last updated: 2026-05-09

Persistent workspace root: `/workspace`

Important persistent paths:
- Code: `/workspace/mmdetection3d`
- Dataset: `/workspace/data/nuscenes`
- Checkpoints: `/workspace/checkpoints`
- Work dirs: `/workspace/work_dirs`
- Outputs/logs: `/workspace/outputs`
- Scripts: `/workspace/scripts`
- Analysis results: `/workspace/analysis_results`
- Miniconda: `/workspace/tools/miniconda3`
- Conda env: `/workspace/envs/bevfusion`

Current environment activation:
```bash
source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion
cd /workspace/mmdetection3d
export PYTHONPATH=/workspace/mmdetection3d:${PYTHONPATH:-}
```

Latest completed stage:
- Stage 18 completed.
- Stage 18 replaced the nearly constant learned feature gate with an
  input-adaptive proxy gate driven by raw sensor reliability metadata.
- Proxy gate diagnostics, max 10 samples:
  clean gate_mean = 0.0531;
  camera fog severe gate_mean = 0.1282;
  LiDAR dropout severe gate_mean = 0.1289.
- Camera fog severe lowered camera reliability from about 0.736 to 0.359.
- LiDAR dropout severe lowered LiDAR reliability from about 0.790 to 0.355.
- Proxy gate clean eval:
  mAP = 0.5598058052618606, NDS = 0.5684444524792993.
- Proxy gate camera fog severe eval:
  mAP = 0.5525624004644998, NDS = 0.5645863065540148.
- Proxy gate LiDAR dropout severe eval:
  mAP = 0.5428999178939835, NDS = 0.5542659516023942.
- Interpretation: reliability proxy/gate responsiveness is now working and
  clean performance is preserved relative to the previous gate6 checkpoint.
  Metric improvement over the original baseline is not established yet; the
  next step is proxy-gate active corruption-aware finetuning.

Key latest files:
- `/workspace/mmdetection3d/projects/BEVFusion/bevfusion/reliability_attention_fusion.py`
- `/workspace/mmdetection3d/projects/BEVFusion/bevfusion/bevfusion.py`
- `/workspace/mmdetection3d/projects/BEVFusion/bevfusion/corruptions.py`
- `/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_attention_feature_gate_regularized_debug_finetune_nus-3d.py`
- `/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_attention_feature_gate_regularized_gate6_debug_finetune_nus-3d.py`
- `/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_attention_feature_gate_regularized_gate6_clean_eval_nus-3d.py`
- `/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_attention_feature_gate_corruption_runtime_nus-3d.py`
- `/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_attention_proxy_gate_corruption_runtime_nus-3d.py`
- `/workspace/scripts/run_bevfusion_mini_attention_regularized_debug_finetune.sh`
- `/workspace/scripts/run_bevfusion_mini_attention_regularized_gate6_debug_finetune.sh`
- `/workspace/scripts/run_bevfusion_mini_regularized_gate6_finetuned_clean_eval.sh`
- `/workspace/scripts/diagnose_attention_gate_residual.py`
- `/workspace/scripts/run_attention_gate_residual_diagnostics.sh`
- `/workspace/scripts/run_attention_proxy_gate_diagnostics.sh`
- `/workspace/analysis_results/stage16_attention_gate_residual_diagnostics_summary.json`
- `/workspace/analysis_results/stage17_regularized_gate6_debug_summary.json`
- `/workspace/analysis_results/stage18_proxy_gate_reliability_summary.json`
- `/workspace/analysis_results/attention_gate_residual_diagnostics.csv`
- `/workspace/analysis_results/attention_gate_residual_diagnostics.json`

Recommended next step:
- Do not start long training yet.
- Run a short proxy-gate active finetune with mixed clean/corrupted batches.
- Keep clean consistency regularization so clean output stays close to baseline.
- Then evaluate clean, camera fog severe, LiDAR dropout severe, and
  camera+LiDAR all severe again.
