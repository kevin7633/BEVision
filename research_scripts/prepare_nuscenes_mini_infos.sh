#!/usr/bin/env bash
set -euo pipefail

source /workspace/tools/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/envs/bevfusion

cd /workspace/mmdetection3d

if [ ! -d /workspace/data/nuscenes/v1.0-mini ]; then
  echo "Missing /workspace/data/nuscenes/v1.0-mini"
  echo "Place nuScenes-mini raw files under /workspace/data/nuscenes first."
  exit 2
fi

python tools/create_data.py nuscenes \
  --root-path /workspace/data/nuscenes \
  --out-dir /workspace/data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini \
  --max-sweeps 10

ls -lh /workspace/data/nuscenes/*infos*.pkl
