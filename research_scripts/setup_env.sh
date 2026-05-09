#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/workspace"
MINICONDA_DIR="${WORKSPACE}/tools/miniconda3"
ENV_DIR="${WORKSPACE}/envs/bevfusion"
CONDA_PKGS_DIR="${WORKSPACE}/tools/conda_pkgs"
PIP_CACHE_DIR="${WORKSPACE}/.cache/pip"
REPO_DIR="${WORKSPACE}/mmdetection3d"
LOG_DIR="${WORKSPACE}/outputs"

mkdir -p "${WORKSPACE}/tools" "${WORKSPACE}/envs" "${CONDA_PKGS_DIR}" "${PIP_CACHE_DIR}" "${LOG_DIR}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

if [ ! -x "${MINICONDA_DIR}/bin/conda" ]; then
  installer="${WORKSPACE}/tools/Miniconda3-latest-Linux-x86_64.sh"
  if [ ! -f "${installer}" ]; then
    curl -L "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "${installer}"
  fi
  bash "${installer}" -b -p "${MINICONDA_DIR}"
fi

source "${MINICONDA_DIR}/etc/profile.d/conda.sh"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  conda create -p "${ENV_DIR}" python=3.10 -y
fi

conda activate "${ENV_DIR}"

echo "[Python before install]"
which python
which pip
python -V
pip --version

python -m pip install "pip==23.3.1" "setuptools<70" wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.1.0 torchvision==0.16.0
python -m pip install "numpy<2"

python -m pip install -U openmim
python -m mim install "mmengine>=0.7.1,<1.0.0"
python -m mim install "mmcv>=2.0.0,<2.2.0"
python -m mim install "mmdet>=3.0.0,<3.3.0"

cd "${REPO_DIR}"
python -m pip install \
  "numpy<2" \
  "opencv-python==4.11.0.86" \
  "opencv-python-headless==4.11.0.86" \
  "nuscenes-devkit" \
  "numba" \
  "plyfile" \
  "scikit-image" \
  "lyft_dataset_sdk" \
  "open3d" \
  "absl-py" \
  "grpcio" \
  "protobuf" \
  "tensorboard-data-server" \
  "spconv-cu118" \
  "ninja"

python -m pip install -e . --no-deps
MAX_JOBS="${MAX_JOBS:-4}" python projects/BEVFusion/setup.py develop

echo "[Import check]"
which python
which pip
python -V
python - <<'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())

for name in ['mmcv', 'mmengine', 'mmdet', 'mmdet3d', 'spconv', 'nuscenes', 'cv2', 'numpy']:
    mod = __import__(name)
    print(f'{name}:', getattr(mod, '__version__', 'unknown'))
PY
