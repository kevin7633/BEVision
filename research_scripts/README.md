# Research Scripts

These scripts reproduce the staged nuScenes-mini BEVFusion robustness workflow.
They assume the RunPod-style persistent layout used in the experiment:

- Code: `/workspace/mmdetection3d`
- Dataset: `/workspace/data/nuscenes`
- Checkpoints: `/workspace/checkpoints`
- Work dirs: `/workspace/work_dirs`
- Outputs: `/workspace/outputs`
- Analysis: `/workspace/analysis_results`
- Conda env: `/workspace/envs/bevfusion`

Large artifacts are not included in git. Downloaded datasets, checkpoints,
logs, rendered outputs, and conda environments should stay outside the repo or
under ignored runtime directories.
