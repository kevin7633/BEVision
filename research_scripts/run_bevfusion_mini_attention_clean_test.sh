#!/usr/bin/env bash
set -euo pipefail

source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion

export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/checkpoints/torch_cache
mkdir -p "${TORCH_HOME}"

mode="${1:-on}"
if [[ "${mode}" == "on" ]]; then
  config="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_attention_nus-3d.py"
  exp_name="bevfusion_mini_attention_clean"
  use_attention_fusion_flag="--use-attention-fusion"
elif [[ "${mode}" == "feature" ]]; then
  config="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_attention_feature_gate_nus-3d.py"
  exp_name="bevfusion_mini_attention_feature_gate_clean"
  use_attention_fusion_flag="--use-attention-fusion"
elif [[ "${mode}" == "off" ]]; then
  config="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_attention_off_nus-3d.py"
  exp_name="bevfusion_mini_attention_off_clean"
  use_attention_fusion_flag=""
else
  echo "mode must be 'on', 'off', or 'feature', got: ${mode}" >&2
  exit 2
fi

checkpoint="/workspace/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
work_dir="/workspace/work_dirs/${exp_name}"
json_prefix="/workspace/outputs/${exp_name}/results"
log_file="/workspace/outputs/${exp_name}.log"

mkdir -p "${work_dir}" "$(dirname "${json_prefix}")"

cd /workspace/mmdetection3d

python tools/test.py "${config}" "${checkpoint}" \
  --work-dir "${work_dir}" \
  --task multi-modality_det \
  --cfg-options \
    test_dataloader.batch_size=1 \
    test_dataloader.num_workers=0 \
    test_dataloader.persistent_workers=False \
    val_dataloader.batch_size=1 \
    val_dataloader.num_workers=0 \
    val_dataloader.persistent_workers=False \
    test_evaluator.jsonfile_prefix="${json_prefix}" \
  2>&1 | tee "${log_file}"

python /workspace/scripts/summarize_bevfusion_results.py \
  --exp-name "${exp_name}" \
  --config "/workspace/mmdetection3d/${config}" \
  --checkpoint "${checkpoint}" \
  --dataset-version v1.0-mini \
  --corruption-type none \
  --corruption-severity none \
  --clean-or-corrupted clean \
  ${use_attention_fusion_flag} \
  --use-self-attention \
  --use-reliability-gate \
  --output-dir "/workspace/outputs/${exp_name}" \
  --metrics "${json_prefix}/pred_instances_3d/metrics_summary.json" \
  --predictions "${json_prefix}/pred_instances_3d/results_nusc.json" \
  --notes "Stage 8 clean preservation check (${mode})" \
  --class-metrics-json "/workspace/analysis_results/${exp_name}_class_metrics.json"
