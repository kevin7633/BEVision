#!/usr/bin/env bash
set -euo pipefail

source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion

export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/checkpoints/torch_cache

CLEAN_CONFIG="/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_attention_feature_gate_nus-3d.py"
CORR_CONFIG="/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_attention_feature_gate_corruption_runtime_nus-3d.py"
CHECKPOINT="${1:-/workspace/work_dirs/bevfusion_mini_attention_feature_gate_debug_finetune/iter_10.pth}"
SUMMARY="/workspace/scripts/summarize_bevfusion_results.py"

run_clean() {
  local exp_name="bevfusion_mini_finetuned_iter10_clean"
  local json_prefix="/workspace/outputs/${exp_name}/results"
  local work_dir="/workspace/work_dirs/${exp_name}"
  local log_path="/workspace/outputs/${exp_name}.log"
  mkdir -p "${work_dir}" "$(dirname "${json_prefix}")"

  echo "=== Running ${exp_name} ==="
  (
    cd /workspace/mmdetection3d
    python tools/test.py "${CLEAN_CONFIG}" "${CHECKPOINT}" \
      --work-dir "${work_dir}" \
      --task multi-modality_det \
      --cfg-options \
        test_evaluator.jsonfile_prefix="${json_prefix}" \
        test_dataloader.batch_size=1 \
        test_dataloader.num_workers=0 \
        test_dataloader.persistent_workers=False
  ) 2>&1 | tee "${log_path}"

  python "${SUMMARY}" \
    --exp-name "${exp_name}" \
    --config "${CLEAN_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-version v1.0-mini \
    --corruption-type none \
    --corruption-severity none \
    --clean-or-corrupted clean \
    --use-attention-fusion \
    --use-self-attention \
    --use-reliability-gate \
    --output-dir "/workspace/outputs/${exp_name}/results/pred_instances_3d" \
    --metrics "${json_prefix}/pred_instances_3d/metrics_summary.json" \
    --predictions "${json_prefix}/pred_instances_3d/results_nusc.json" \
    --notes "Stage 12 explicit eval of 10-iter attention-only finetuned checkpoint." \
    --class-metrics-json "/workspace/analysis_results/${exp_name}_class_metrics.json"
}

run_corrupt() {
  local exp_name="$1"
  local cam_corr="$2"
  local lidar_corr="$3"
  local severity="$4"
  local corr_type="$5"
  local log_path="/workspace/outputs/${exp_name}.log"
  mkdir -p /workspace/outputs

  echo "=== Running ${exp_name} ==="
  (
    cd /workspace/mmdetection3d
    BEVFUSION_EXP_NAME="${exp_name}" \
    BEVFUSION_CAMERA_CORRUPTION="${cam_corr}" \
    BEVFUSION_LIDAR_CORRUPTION="${lidar_corr}" \
    BEVFUSION_CORRUPTION_SEVERITY="${severity}" \
    BEVFUSION_CORRUPTION_SEED=2026 \
    python tools/test.py "${CORR_CONFIG}" "${CHECKPOINT}" --launcher none
  ) 2>&1 | tee "${log_path}"

  python "${SUMMARY}" \
    --exp-name "${exp_name}" \
    --config "${CORR_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-version v1.0-mini \
    --corruption-type "${corr_type}" \
    --corruption-severity "${severity}" \
    --clean-or-corrupted corrupted \
    --use-attention-fusion \
    --use-self-attention \
    --use-reliability-gate \
    --output-dir "/workspace/outputs/${exp_name}/results/pred_instances_3d" \
    --metrics "/workspace/outputs/${exp_name}/results/pred_instances_3d/metrics_summary.json" \
    --predictions "/workspace/outputs/${exp_name}/results/pred_instances_3d/results_nusc.json" \
    --notes "Stage 12 explicit eval of 10-iter attention-only finetuned checkpoint." \
    --class-metrics-json "/workspace/analysis_results/${exp_name}_class_metrics.json"
}

run_clean
run_corrupt "bevfusion_mini_finetuned_iter10_camera_fog_severe" "fog" "none" "severe" "camera_fog"
run_corrupt "bevfusion_mini_finetuned_iter10_lidar_dropout_severe" "none" "random_dropout" "severe" "lidar_random_dropout"
run_corrupt "bevfusion_mini_finetuned_iter10_camera_lidar_all_severe" "all" "all" "severe" "camera_lidar_all"
