## BEVision Research Branch

This branch contains a nuScenes-mini research workflow for:

**Reliability-Guided Cross-Self Attention BEV Fusion for robust Camera-LiDAR
BEVFusion under adverse weather and sensor degradation.**

The goal of this branch is not to reproduce full nuScenes SOTA yet. It is a
compact, reproducible research branch for:

- running a BEVFusion baseline on nuScenes-mini or a small subset,
- applying controlled camera and LiDAR corruptions without overwriting the
  original dataset,
- comparing clean and corrupted performance,
- computing lightweight camera/LiDAR reliability proxies,
- adding a residual attention fusion module that reacts more strongly when
  sensor reliability drops,
- recording enough CSV/JSON summaries for follow-up experiments and website or
  dashboard visualization.

Large runtime artifacts are intentionally not committed. The repository does
not include nuScenes data, checkpoints, conda environments, `work_dirs`,
generated outputs, logs, or rendered images.

### What Was Added

Core BEVFusion changes:

- `projects/BEVFusion/bevfusion/corruptions.py`
  - Camera corruption transforms: fog, low-light, contrast reduction, blur,
    noise, occlusion, and combined corruption.
  - LiDAR corruption transforms: random dropout, distance-based dropout,
    intensity noise, noisy points, and combined corruption.
  - `BEVFusionSensorReliabilityProxy`, which computes raw camera and LiDAR
    reliability proxies before model preprocessing.

- `projects/BEVFusion/bevfusion/reliability_attention_fusion.py`
  - `ReliabilityGuidedAttentionResidualFusion`.
  - Implements residual correction:
    `F_final = F_base + gate * A`.
  - Supports scalar, feature-based, and proxy-based reliability gates.
  - Uses cross-attention from LiDAR BEV queries to camera BEV keys/values, then
    optional BEV self-attention on downsampled BEV tokens.

- `projects/BEVFusion/bevfusion/bevfusion.py`
  - Adds optional `attention_fusion_layer`.
  - Keeps baseline behavior when the attention module is disabled.
  - Passes reliability proxy metadata into the residual fusion module.
  - Supports attention-only debug finetuning.

Experiment configs:

- `projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-mini_nus-3d.py`
  - nuScenes-mini baseline config.

- `projects/BEVFusion/configs/bevfusion_mini_corruption_runtime_nus-3d.py`
  - Runtime corruption config controlled through environment variables.

- `projects/BEVFusion/configs/bevfusion_mini_attention_corruption_runtime_nus-3d.py`
  - Baseline plus attention residual module under corruptions.

- `projects/BEVFusion/configs/bevfusion_mini_attention_feature_gate_corruption_runtime_nus-3d.py`
  - Feature-gated attention experiment config.

- `projects/BEVFusion/configs/bevfusion_mini_attention_proxy_gate_corruption_runtime_nus-3d.py`
  - Latest proxy reliability gate config.
  - This is the most important config for the current research direction.

Research scripts:

- `research_scripts/check_stage0_env.sh`
  - Checks storage, conda, CUDA, PyTorch, MMDetection3D packages, GPU, dataset,
    and checkpoint state.

- `research_scripts/run_bevfusion_mini_test.sh`
  - Runs the clean mini baseline test.

- `research_scripts/run_bevfusion_mini_robustness_matrix.sh`
  - Runs clean, camera corruption, LiDAR corruption, and camera+LiDAR
    corruption conditions.

- `research_scripts/verify_corruption_effect.py`
  - Verifies that corruptions are actually changing image/point statistics.

- `research_scripts/compute_reliability_proxies.py`
  - Computes reliability proxy samples and summary tables.

- `research_scripts/diagnose_attention_gate_residual.py`
  - Measures gate values, residual magnitude, and reliability reaction.

- `research_scripts/run_attention_proxy_gate_diagnostics.sh`
  - Runs the latest proxy gate diagnostic set.

- `research_scripts/summarize_bevfusion_results.py`
  - Converts MMDetection3D output JSON files into compact experiment summary
    rows.

Result summaries:

- `research_results/stage18_proxy_gate_reliability_summary.json`
  - Latest stage summary.
  - Shows that the proxy gate now responds to corruption:
    clean gate mean about `0.053`, camera fog severe about `0.128`, and LiDAR
    dropout severe about `0.129`.

- `research_results/experiment_summary.csv`
  - Main table for clean/corrupted metrics, confidence statistics, detection
    counts, configs, checkpoints, and output paths.

- `research_results/stage5_robustness_matrix.csv`
  - Baseline robustness matrix across corruption type and severity.

- `research_results/stage6_reliability_vs_performance.csv`
  - Reliability proxy summaries joined with performance drops.

- `research_results/attention_gate_residual_diagnostics.csv`
  - Gate and residual diagnostics for attention experiments.

- `research_results/resume_notes.md`
  - Practical notes for continuing the RunPod experiment later.

### Current Research Status

The project has reached the first useful mechanism checkpoint:

- Baseline BEVFusion runs on nuScenes-mini.
- Synthetic camera/LiDAR corruptions are applied in-memory, without modifying
  the original data.
- Clean and corrupted results are summarized as CSV/JSON.
- Reliability proxy calculation is implemented.
- Reliability-guided residual attention fusion is implemented.
- The latest proxy gate is input-adaptive:
  - clean gate mean: about `0.053`
  - severe camera fog gate mean: about `0.128`
  - severe LiDAR dropout gate mean: about `0.129`

The current result should be interpreted carefully. The mechanism is now
working and clean performance is preserved relative to the latest attention
checkpoint, but the branch has not yet proven a final robustness gain over the
original baseline. The next research step is short corruption-aware finetuning
with the proxy gate active.

### Recommended Reading Order

For a researcher:

1. `research_results/stage18_proxy_gate_reliability_summary.json`
2. `research_results/experiment_summary.csv`
3. `projects/BEVFusion/bevfusion/reliability_attention_fusion.py`
4. `projects/BEVFusion/bevfusion/corruptions.py`
5. `projects/BEVFusion/configs/bevfusion_mini_attention_proxy_gate_corruption_runtime_nus-3d.py`
6. `research_scripts/run_attention_proxy_gate_diagnostics.sh`

For someone building a website or dashboard:

1. `research_results/experiment_summary.csv`
   - Use this as the main experiment table.
   - Useful fields: `exp_name`, `corruption_type`, `corruption_severity`,
     `mAP`, `NDS`, `mean_confidence`, `detection_count`, `notes`.

2. `research_results/stage5_robustness_matrix.csv`
   - Use this for clean vs corruption severity plots.
   - Suggested charts: mAP by corruption type/severity, NDS by corruption
     type/severity, detection count by condition.

3. `research_results/stage6_reliability_vs_performance.csv`
   - Use this for reliability vs performance-drop visualizations.
   - Suggested charts: camera reliability vs delta mAP, LiDAR reliability vs
     delta NDS, degradation score vs mAP.

4. `research_results/attention_gate_residual_diagnostics.csv`
   - Use this for attention/gate behavior visualizations.
   - Suggested charts: gate mean by condition, residual-to-base ratio by
     condition, reliability mean by condition.

5. `research_results/stage18_proxy_gate_reliability_summary.json`
   - Use this for a compact "latest status" panel.
   - It includes the latest interpretation, key metrics, and next steps.

### How To Continue Experiments

Activate the RunPod workspace environment:

```bash
source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion
cd /workspace/mmdetection3d
export PYTHONPATH=/workspace/mmdetection3d:${PYTHONPATH:-}
```

Run the latest proxy gate diagnostics:

```bash
bash research_scripts/run_attention_proxy_gate_diagnostics.sh
```

Run a clean mini baseline:

```bash
bash research_scripts/run_bevfusion_mini_test.sh
```

Run the robustness matrix:

```bash
bash research_scripts/run_bevfusion_mini_robustness_matrix.sh
```

Runtime artifacts should stay outside git, under the persistent RunPod
workspace paths:

- `/workspace/data/nuscenes`
- `/workspace/checkpoints`
- `/workspace/work_dirs`
- `/workspace/outputs`
- `/workspace/analysis_results`
- `/workspace/envs`
- `/workspace/tools`
