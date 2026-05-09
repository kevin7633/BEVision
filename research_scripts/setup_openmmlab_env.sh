#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mmdetection3d

python -m pip install -U openmim
python -m pip install "numpy<2" "blinker>=1.9.0" --ignore-installed blinker
python -m mim install "mmengine>=0.7.1,<1.0.0"
python -m mim install "mmcv>=2.0.0,<2.2.0"
python -m mim install "mmdet>=3.0.0,<3.3.0"
python -m pip install -v -e .
python -m pip install \
  "numpy<2" \
  "opencv-python==4.11.0.86" \
  "opencv-python-headless==4.11.0.86" \
  "nuscenes-devkit" \
  "spconv-cu118" \
  "ninja"
MAX_JOBS="${MAX_JOBS:-4}" python projects/BEVFusion/setup.py develop
