#!/usr/bin/env bash
set -euo pipefail

cd /workspace

echo "[Workspace]"
find /workspace -maxdepth 3 -type d | sort

echo
echo "[Disk]"
df -h /workspace

echo
echo "[GPU]"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader || true

echo
echo "[Python Packages]"
python - <<'PY'
import importlib
import sys

print("python_executable:", sys.executable)
print("python_version:", sys.version.replace("\n", " "))

modules = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("mmcv", "mmcv"),
    ("mmengine", "mmengine"),
    ("mmdet", "mmdet"),
    ("mmdet3d", "mmdet3d"),
    ("spconv", "spconv"),
    ("nuscenes-devkit", "nuscenes"),
    ("opencv", "cv2"),
    ("numpy", "numpy"),
]

for label, modname in modules:
    try:
        mod = importlib.import_module(modname)
        print(f"{label}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{label}: NOT_IMPORTABLE ({type(exc).__name__}: {exc})")

try:
    import torch
    print("torch_cuda_available:", torch.cuda.is_available())
    print("torch_cuda_version:", torch.version.cuda)
    print("torch_cudnn_version:", torch.backends.cudnn.version())
    print("torch_gpu_count:", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(f"torch_gpu_{idx}: {props.name}, total_memory_GB={props.total_memory / (1024 ** 3):.2f}")
except Exception as exc:
    print(f"torch_cuda_probe_failed: {type(exc).__name__}: {exc}")
PY

echo
echo "[nuScenes]"
find /workspace/data/nuscenes -maxdepth 3 -type f | sort | sed -n '1,80p' || true

echo
echo "[Checkpoints]"
find /workspace/checkpoints -maxdepth 3 \( -iname '*.pth' -o -iname '*.ckpt' \) | sort || true
