# Reliability-Guided BEVFusion Mini Experiments

This folder contains lightweight, shareable summaries for the nuScenes-mini
robustness experiments. It intentionally excludes raw nuScenes data,
checkpoints, conda environments, work directories, rendered outputs, and logs.

Key files:

- `resume_notes.md`: persistent notes for continuing the RunPod experiment.
- `stage18_proxy_gate_reliability_summary.json`: latest proxy reliability gate
  summary.
- `experiment_summary.csv` / `experiment_summary.json`: compact experiment
  table with metrics and output paths from the original `/workspace` run.
- `attention_gate_residual_diagnostics.csv`: gate/residual diagnostic table.

The scripts used to reproduce the mini experiments are in `research_scripts/`.
