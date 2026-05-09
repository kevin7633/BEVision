#!/usr/bin/env bash
set -euo pipefail

source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion

export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/checkpoints/torch_cache

CONFIG="/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_mini_attention_corruption_runtime_nus-3d.py"
CHECKPOINT="/workspace/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
SUMMARY="/workspace/scripts/summarize_bevfusion_results.py"

run_one() {
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
    python tools/test.py "${CONFIG}" "${CHECKPOINT}" --launcher none
  ) 2>&1 | tee "${log_path}"

  python "${SUMMARY}" \
    --exp-name "${exp_name}" \
    --config "${CONFIG}" \
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
    --notes "Stage 9 pre-finetune attention check; zero-initialized residual branch." \
    --class-metrics-json "/workspace/analysis_results/${exp_name}_class_metrics.json"
}

run_one "bevfusion_mini_attention_camera_fog_severe" "fog" "none" "severe" "camera_fog"
run_one "bevfusion_mini_attention_lidar_dropout_severe" "none" "random_dropout" "severe" "lidar_random_dropout"
run_one "bevfusion_mini_attention_camera_lidar_all_severe" "all" "all" "severe" "camera_lidar_all"
