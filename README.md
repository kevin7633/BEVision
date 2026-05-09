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

---

<div align="center">
  <img src="resources/mmdet3d-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmdet3d)](https://pypi.org/project/mmdet3d)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection3d.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection3d/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection3d/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection3d/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection3d)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection3d.svg)](https://github.com/open-mmlab/mmdetection3d/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdetection3d.svg)](https://github.com/open-mmlab/mmdetection3d/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmdetection3d.svg)](https://github.com/open-mmlab/mmdetection3d/issues)

[📘Documentation](https://mmdetection3d.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) |
[👀Model Zoo](https://mmdetection3d.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmdetection3d.readthedocs.io/en/latest/notes/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmdetection3d/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmdetection3d/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1046608014234370059" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## Introduction

MMDetection3D is an open source object detection toolbox based on PyTorch, towards the next-generation platform for general 3D detection. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The main branch works with **PyTorch 1.8+**.

![demo image](resources/mmdet3d_outdoor_demo.gif)

<details open>
<summary>Major features</summary>

- **Support multi-modality/single-modality detectors out of box**

  It directly supports multi-modality/single-modality detectors including MVXNet, VoteNet, PointPillars, etc.

- **Support indoor/outdoor 3D detection out of box**

  It directly supports popular indoor and outdoor 3D detection datasets, including ScanNet, SUNRGB-D, Waymo, nuScenes, Lyft, and KITTI. For nuScenes dataset, we also support [nuImages dataset](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/nuimages).

- **Natural integration with 2D detection**

  All the about **300+ models, methods of 40+ papers**, and modules supported in [MMDetection](https://github.com/open-mmlab/mmdetection/blob/3.x/docs/en/model_zoo.md) can be trained or used in this codebase.

- **High efficiency**

  It trains faster than other codebases. The main results are as below. Details can be found in [benchmark.md](./docs/en/notes/benchmarks.md). We compare the number of samples trained per second (the higher, the better). The models that are not supported by other codebases are marked by `✗`.

  |       Methods       | MMDetection3D | [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | [votenet](https://github.com/facebookresearch/votenet) | [Det3D](https://github.com/poodarchu/Det3D) |
  | :-----------------: | :-----------: | :--------------------------------------------------: | :----------------------------------------------------: | :-----------------------------------------: |
  |       VoteNet       |      358      |                          ✗                           |                           77                           |                      ✗                      |
  |  PointPillars-car   |      141      |                          ✗                           |                           ✗                            |                     140                     |
  | PointPillars-3class |      107      |                          44                          |                           ✗                            |                      ✗                      |
  |       SECOND        |      40       |                          30                          |                           ✗                            |                      ✗                      |
  |       Part-A2       |      17       |                          14                          |                           ✗                            |                      ✗                      |

</details>

Like [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMCV](https://github.com/open-mmlab/mmcv), MMDetection3D can also be used as a library to support different projects on top of it.

## What's New

### Highlight

In version 1.4, MMDetecion3D refactors the Waymo dataset and accelerates the preprocessing, training/testing setup, and evaluation of Waymo dataset. We also extends the support for camera-based, such as Monocular and BEV, 3D object detection models on Waymo. A detailed description of the Waymo data information is provided [here](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html).

Besides, in version 1.4, MMDetection3D provides [Waymo-mini](https://download.openmmlab.com/mmdetection3d/data/waymo_mmdet3d_after_1x4/waymo_mini.tar.gz) to help community users get started with Waymo and use it for quick iterative development.

**v1.4.0** was released in 8/1/2024：

- Support the training of [DSVT](<(https://arxiv.org/abs/2301.06051)>) in `projects`
- Support [Nerf-Det](https://arxiv.org/abs/2307.14620) in `projects`
- Refactor Waymo dataset

**v1.3.0** was released in 18/10/2023:

- Support [CENet](https://arxiv.org/abs/2207.12691) in `projects`
- Enhance demos with new 3D inferencers

**v1.2.0** was released in 4/7/2023

- Support [New Config Type](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta) in `mmdet3d/configs`
- Support the inference of [DSVT](<(https://arxiv.org/abs/2301.06051)>) in `projects`
- Support downloading datasets from [OpenDataLab](https://opendatalab.com/) using `mim`

**v1.1.1** was released in 30/5/2023:

- Support [TPVFormer](https://arxiv.org/pdf/2302.07817.pdf) in `projects`
- Support the training of BEVFusion in `projects`
- Support lidar-based 3D semantic segmentation benchmark

## Installation

Please refer to [Installation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) for installation instructions.

## Getting Started

For detailed user guides and advanced guides, please refer to our [documentation](https://mmdetection3d.readthedocs.io/en/latest/):

<details>
<summary>User Guides</summary>

- [Train & Test](https://mmdetection3d.readthedocs.io/en/latest/user_guides/index.html#train-test)
  - [Learn about Configs](https://mmdetection3d.readthedocs.io/en/latest/user_guides/config.html)
  - [Coordinate System](https://mmdetection3d.readthedocs.io/en/latest/user_guides/coord_sys_tutorial.html)
  - [Dataset Preparation](https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html)
  - [Customize Data Pipelines](https://mmdetection3d.readthedocs.io/en/latest/user_guides/data_pipeline.html)
  - [Test and Train on Standard Datasets](https://mmdetection3d.readthedocs.io/en/latest/user_guides/train_test.html)
  - [Inference](https://mmdetection3d.readthedocs.io/en/latest/user_guides/inference.html)
  - [Train with Customized Datasets](https://mmdetection3d.readthedocs.io/en/latest/user_guides/new_data_model.html)
- [Useful Tools](https://mmdetection3d.readthedocs.io/en/latest/user_guides/index.html#useful-tools)

</details>

<details>
<summary>Advanced Guides</summary>

- [Datasets](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/index.html#datasets)
  - [KITTI Dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/kitti.html)
  - [NuScenes Dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html)
  - [Lyft Dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/lyft.html)
  - [Waymo Dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html)
  - [SUN RGB-D Dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/sunrgbd.html)
  - [ScanNet Dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/scannet.html)
  - [S3DIS Dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/s3dis.html)
  - [SemanticKITTI Dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/semantickitti.html)
- [Supported Tasks](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/index.html#supported-tasks)
  - [LiDAR-Based 3D Detection](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/supported_tasks/lidar_det3d.html)
  - [Vision-Based 3D Detection](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/supported_tasks/vision_det3d.html)
  - [LiDAR-Based 3D Semantic Segmentation](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/supported_tasks/lidar_sem_seg3d.html)
- [Customization](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/index.html#customization)
  - [Customize Datasets](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/customize_dataset.html)
  - [Customize Models](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/customize_models.html)
  - [Customize Runtime Settings](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/customize_runtime.html)

</details>

## Overview of Benchmark and Model Zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

<div align="center">
  <b>Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Heads</b>
      </td>
      <td>
        <b>Features</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li><a href="configs/pointnet2">PointNet (CVPR'2017)</a></li>
        <li><a href="configs/pointnet2">PointNet++ (NeurIPS'2017)</a></li>
        <li><a href="configs/regnet">RegNet (CVPR'2020)</a></li>
        <li><a href="configs/dgcnn">DGCNN (TOG'2019)</a></li>
        <li>DLA (CVPR'2018)</li>
        <li>MinkResNet (CVPR'2019)</li>
        <li><a href="configs/minkunet">MinkUNet (CVPR'2019)</a></li>
        <li><a href="configs/cylinder3d">Cylinder3D (CVPR'2021)</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="configs/free_anchor">FreeAnchor (NeurIPS'2019)</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="configs/dynamic_voxelization">Dynamic Voxelization (CoRL'2019)</a></li>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="middle">
      <td>
        <b>LiDAR-based 3D Object Detection</b>
      </td>
      <td>
        <b>Camera-based 3D Object Detection</b>
      </td>
      <td>
        <b>Multi-modal 3D Object Detection</b>
      </td>
      <td>
        <b>3D Semantic Segmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <li><b>Outdoor</b></li>
        <ul>
            <li><a href="configs/second">SECOND (Sensor'2018)</a></li>
            <li><a href="configs/pointpillars">PointPillars (CVPR'2019)</a></li>
            <li><a href="configs/ssn">SSN (ECCV'2020)</a></li>
            <li><a href="configs/3dssd">3DSSD (CVPR'2020)</a></li>
            <li><a href="configs/sassd">SA-SSD (CVPR'2020)</a></li>
            <li><a href="configs/point_rcnn">PointRCNN (CVPR'2019)</a></li>
            <li><a href="configs/parta2">Part-A2 (TPAMI'2020)</a></li>
            <li><a href="configs/centerpoint">CenterPoint (CVPR'2021)</a></li>
            <li><a href="configs/pv_rcnn">PV-RCNN (CVPR'2020)</a></li>
            <li><a href="projects/CenterFormer">CenterFormer (ECCV'2022)</a></li>
        </ul>
        <li><b>Indoor</b></li>
        <ul>
            <li><a href="configs/votenet">VoteNet (ICCV'2019)</a></li>
            <li><a href="configs/h3dnet">H3DNet (ECCV'2020)</a></li>
            <li><a href="configs/groupfree3d">Group-Free-3D (ICCV'2021)</a></li>
            <li><a href="configs/fcaf3d">FCAF3D (ECCV'2022)</a></li>
            <li><a href="projects/TR3D">TR3D (ArXiv'2023)</a></li>
      </ul>
      </td>
      <td>
        <li><b>Outdoor</b></li>
        <ul>
          <li><a href="configs/imvoxelnet">ImVoxelNet (WACV'2022)</a></li>
          <li><a href="configs/smoke">SMOKE (CVPRW'2020)</a></li>
          <li><a href="configs/fcos3d">FCOS3D (ICCVW'2021)</a></li>
          <li><a href="configs/pgd">PGD (CoRL'2021)</a></li>
          <li><a href="configs/monoflex">MonoFlex (CVPR'2021)</a></li>
          <li><a href="projects/DETR3D">DETR3D (CoRL'2021)</a></li>
          <li><a href="projects/PETR">PETR (ECCV'2022)</a></li>
        </ul>
        <li><b>Indoor</b></li>
        <ul>
          <li><a href="configs/imvoxelnet">ImVoxelNet (WACV'2022)</a></li>
        </ul>
      </td>
      <td>
        <li><b>Outdoor</b></li>
        <ul>
          <li><a href="configs/mvxnet">MVXNet (ICRA'2019)</a></li>
          <li><a href="projects/BEVFusion">BEVFusion (ICRA'2023)</a></li>
        </ul>
        <li><b>Indoor</b></li>
        <ul>
          <li><a href="configs/imvotenet">ImVoteNet (CVPR'2020)</a></li>
        </ul>
      </td>
      <td>
        <li><b>Outdoor</b></li>
        <ul>
          <li><a href="configs/minkunet">MinkUNet (CVPR'2019)</a></li>
          <li><a href="configs/spvcnn">SPVCNN (ECCV'2020)</a></li>
          <li><a href="configs/cylinder3d">Cylinder3D (CVPR'2021)</a></li>
          <li><a href="projects/TPVFormer">TPVFormer (CVPR'2023)</a></li>
        </ul>
        <li><b>Indoor</b></li>
        <ul>
          <li><a href="configs/pointnet2">PointNet++ (NeurIPS'2017)</a></li>
          <li><a href="configs/paconv">PAConv (CVPR'2021)</a></li>
          <li><a href="configs/dgcnn">DGCNN (TOG'2019)</a></li>
        </ul>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

|               | ResNet | VoVNet | Swin-T | PointNet++ | SECOND | DGCNN | RegNetX | DLA | MinkResNet | Cylinder3D | MinkUNet |
| :-----------: | :----: | :----: | :----: | :--------: | :----: | :---: | :-----: | :-: | :--------: | :--------: | :------: |
|    SECOND     |   ✗    |   ✗    |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
| PointPillars  |   ✗    |   ✗    |   ✗    |     ✗      |   ✓    |   ✗   |    ✓    |  ✗  |     ✗      |     ✗      |    ✗     |
|  FreeAnchor   |   ✗    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✓    |  ✗  |     ✗      |     ✗      |    ✗     |
|    VoteNet    |   ✗    |   ✗    |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|    H3DNet     |   ✗    |   ✗    |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|     3DSSD     |   ✗    |   ✗    |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|    Part-A2    |   ✗    |   ✗    |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|    MVXNet     |   ✓    |   ✗    |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|  CenterPoint  |   ✗    |   ✗    |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|      SSN      |   ✗    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✓    |  ✗  |     ✗      |     ✗      |    ✗     |
|   ImVoteNet   |   ✓    |   ✗    |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|    FCOS3D     |   ✓    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|  PointNet++   |   ✗    |   ✗    |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
| Group-Free-3D |   ✗    |   ✗    |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|  ImVoxelNet   |   ✓    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|    PAConv     |   ✗    |   ✗    |   ✗    |     ✓      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|     DGCNN     |   ✗    |   ✗    |   ✗    |     ✗      |   ✗    |   ✓   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|     SMOKE     |   ✗    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✓  |     ✗      |     ✗      |    ✗     |
|      PGD      |   ✓    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|   MonoFlex    |   ✗    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✓  |     ✗      |     ✗      |    ✗     |
|    SA-SSD     |   ✗    |   ✗    |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|    FCAF3D     |   ✗    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✓      |     ✗      |    ✗     |
|    PV-RCNN    |   ✗    |   ✗    |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|  Cylinder3D   |   ✗    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✓      |    ✗     |
|   MinkUNet    |   ✗    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✓     |
|    SPVCNN     |   ✗    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✓     |
|   BEVFusion   |   ✗    |   ✗    |   ✓    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
| CenterFormer  |   ✗    |   ✗    |   ✗    |     ✗      |   ✓    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|     TR3D      |   ✗    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✓      |     ✗      |    ✗     |
|    DETR3D     |   ✓    |   ✓    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|     PETR      |   ✗    |   ✓    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |
|   TPVFormer   |   ✓    |   ✗    |   ✗    |     ✗      |   ✗    |   ✗   |    ✗    |  ✗  |     ✗      |     ✗      |    ✗     |

**Note:** All the about **500+ models, methods of 90+ papers** in 2D detection supported by [MMDetection](https://github.com/open-mmlab/mmdetection/blob/3.x/docs/en/model_zoo.md) can be trained or used in this codebase.

## FAQ

Please refer to [FAQ](docs/en/notes/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMDetection3D. Please refer to [CONTRIBUTING.md](docs/en/notes/contribution_guides.md) for the contributing guideline.

## Acknowledgement

MMDetection3D is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new 3D detectors.

## Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{mmdet3d2020,
    title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},
    author={MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMEval](https://github.com/open-mmlab/mmeval): A unified evaluation library for multiple machine learning libraries.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab pre-training toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMagic](https://github.com/open-mmlab/mmagic): Open**MM**Lab **A**dvanced, **G**enerative and **I**ntelligent **C**reation toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
