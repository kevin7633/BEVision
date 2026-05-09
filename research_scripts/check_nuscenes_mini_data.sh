#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/data/nuscenes"

echo "[nuScenes mini data check]"
echo "root: ${ROOT}"

required_dirs=(
  "${ROOT}/v1.0-mini"
  "${ROOT}/samples"
  "${ROOT}/sweeps"
  "${ROOT}/maps"
)

missing=0
for path in "${required_dirs[@]}"; do
  if [ -e "${path}" ]; then
    echo "OK: ${path}"
  else
    echo "MISSING: ${path}"
    missing=1
  fi
done

echo
echo "[Existing info files]"
find "${ROOT}" -maxdepth 1 -type f -name '*infos*.pkl' -print | sort || true

echo
echo "[Size]"
du -sh "${ROOT}" || true

if [ "${missing}" -ne 0 ]; then
  echo
  echo "nuScenes-mini raw files are incomplete."
  echo "Expected layout under /workspace/data/nuscenes:"
  echo "  v1.0-mini/*.json"
  echo "  samples/{CAM_*,LIDAR_TOP}/..."
  echo "  sweeps/..."
  echo "  maps/..."
  exit 2
fi

echo
echo "nuScenes-mini raw files appear to be present."
