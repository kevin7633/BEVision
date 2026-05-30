# Reliability-Guided BEVFusion

이 브랜치는 nuScenes Camera-LiDAR BEVFusion에 **sensor reliability 기반 residual fusion**을 추가한 실험 코드입니다. 목표는 clean validation 성능만 무리하게 올리는 것이 아니라, 카메라/라이다 품질이 낮아지는 상황에서 fusion이 덜 무너지도록 만드는 것입니다.

대용량 파일은 저장소에 포함하지 않습니다. nuScenes 데이터, 체크포인트, conda 환경, `work_dirs/`, 로그 파일은 각자 로컬/서버 경로에 두고 실행합니다.

## 핵심 구조

기본 BEVFusion은 camera BEV feature와 LiDAR BEV feature를 `ConvFuser`로 합칩니다. 이 브랜치의 reliability 모델은 그 뒤에 residual layer를 추가합니다.

```text
fused_feature = base_feature + gate * correction
```

기본 설정은 다음과 같습니다.

- `base_source='lidar'`: LiDAR BEV feature를 안정적인 base로 사용합니다.
- `correction_source='camera_lidar'`: camera+LiDAR BEV feature에서 보정량을 예측합니다.
- `gate_mode='image_lidar'`: image reliability와 LiDAR reliability proxy로 gate를 계산합니다.
- `gate_max=0.75`: 보정량이 base를 과하게 덮지 않도록 gate를 제한합니다.
- correction 마지막 conv는 zero-init이라 학습 초기는 base feature에서 출발합니다.

관련 구현 파일:

- `projects/BEVFusion/bevfusion/reliability_fusion.py`
- `projects/BEVFusion/bevfusion/corruptions.py`
- `projects/BEVFusion/bevfusion/bevfusion.py`

## 포함된 config

주요 config는 `configs/custom/` 아래에 있습니다.

- `bevfusion_baseline_1xb1_nuscenes.py`: full trainval baseline
- `bevfusion_reliability_1xb1_nuscenes.py`: reliability residual fusion 모델
- `bevfusion_baseline_debug_iters.py`: 짧은 baseline smoke test
- `bevfusion_reliability_debug_iters.py`: 짧은 reliability smoke test
- `bevfusion_ablation_residual_no_reliability.py`: reliability gate 없는 residual ablation
- `bevfusion_ablation_image_gate.py`: image reliability gate만 사용
- `bevfusion_ablation_lidar_gate.py`: LiDAR reliability gate만 사용
- `bevfusion_ablation_attention_reliability.py`: channel/spatial attention ablation
- `bevfusion_corruption_runtime_eval.py`: validation 시점 corruption 평가

## 환경 설치

권장 환경은 Python 3.9, CUDA 11.8, PyTorch 2.1.x, MMCV 2.1.x, MMEngine 0.10.x, MMDetection 3.2.x, MMDetection3D 1.4.x 계열입니다.

예시:

```bash
conda create -n bevfusion python=3.9 -y
conda activate bevfusion

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install "mmengine==0.10.7"
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
pip install -v -e .
pip install -v -e projects/BEVFusion
```

서버 CUDA 경로가 필요하면 먼저 지정합니다.

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
```

## 데이터 준비

기본 데이터 위치는 `data/nuscenes/`입니다. 다른 위치를 쓰려면 `NUSCENES_DATA_ROOT`를 지정하세요.

```bash
export NUSCENES_DATA_ROOT=/path/to/nuscenes/
```

필요한 구조:

```text
$NUSCENES_DATA_ROOT/
├── v1.0-trainval/
├── samples/
├── sweeps/
├── maps/
├── nuscenes_infos_train.pkl
└── nuscenes_infos_val.pkl
```

raw nuScenes는 있는데 info pkl이 없다면:

```bash
python tools/custom/create_nuscenes_trainval_infos.py \
  --root-path "$NUSCENES_DATA_ROOT" \
  --out-dir "$NUSCENES_DATA_ROOT" \
  --extra-tag nuscenes \
  --version v1.0-trainval \
  --max-sweeps 10
```

데이터 레이아웃 확인:

```bash
python tools/custom/check_nuscenes_layout.py --data-root "$NUSCENES_DATA_ROOT"
```

## 실행 전 sanity check

먼저 config/model build와 dataset sample을 확인합니다.

```bash
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export NUSCENES_DATA_ROOT=/path/to/nuscenes/
export BEVFUSION_WORK_ROOT=/path/to/work_dirs

python tools/custom/run_config_sanity.py \
  configs/custom/bevfusion_baseline_1xb1_nuscenes.py \
  --build-model --skip-init-cfg

python tools/custom/run_config_sanity.py \
  configs/custom/bevfusion_reliability_1xb1_nuscenes.py \
  --build-model --skip-init-cfg

python tools/custom/check_dataset_sample.py \
  configs/custom/bevfusion_reliability_1xb1_nuscenes.py \
  --split train --index 0
```

짧은 100 iteration smoke test:

```bash
BEVFUSION_DEBUG_MAX_ITERS=100 \
BEVFUSION_LOG_INTERVAL=10 \
BEVFUSION_EXP_NAME=reliability_debug_100iter \
python tools/train.py configs/custom/bevfusion_reliability_debug_iters.py
```

## Full Train

V100 32GB 1장 기준으로 AMP에서 `grad_norm: nan`이 날 수 있어, 기본은 non-AMP입니다. batch size는 서버 메모리에 맞춰 조정하세요.

Baseline:

```bash
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export NUSCENES_DATA_ROOT=/path/to/nuscenes/
export BEVFUSION_WORK_ROOT=/path/to/work_dirs
export BEVFUSION_BATCH_SIZE=1
export BEVFUSION_NUM_WORKERS=2
export BEVFUSION_LOG_INTERVAL=20
export BEVFUSION_EXP_NAME=baseline_bevfusion_full_trainval

python tools/train.py configs/custom/bevfusion_baseline_1xb1_nuscenes.py
```

Reliability 모델:

```bash
export BEVFUSION_EXP_NAME=reliability_bevfusion_full_trainval
python tools/train.py configs/custom/bevfusion_reliability_1xb1_nuscenes.py
```

두 실험을 순차 실행하려면:

```bash
bash tools/custom/run_full_train_sequence.sh
```

## 평가

Clean validation:

```bash
python tools/test.py \
  configs/custom/bevfusion_reliability_1xb1_nuscenes.py \
  /path/to/checkpoint.pth \
  --work-dir "$BEVFUSION_WORK_ROOT/eval_clean"
```

Corruption matrix 명령만 출력:

```bash
python tools/custom/run_corruption_matrix.py \
  --checkpoint /path/to/checkpoint.pth \
  --severity moderate
```

실제 순차 실행:

```bash
python tools/custom/run_corruption_matrix.py \
  --checkpoint /path/to/checkpoint.pth \
  --severity moderate \
  --run
```

gate 통계 추출:

```bash
python tools/custom/collect_gate_stats.py \
  "$BEVFUSION_WORK_ROOT"/reliability_bevfusion_full_trainval/*.log \
  --out "$BEVFUSION_WORK_ROOT/reliability_gate_stats.csv"
```

## 예상 시간

이 설정은 `max_epochs=6`입니다. V100 32GB 1장, non-AMP, batch size 2 기준 기존 측정에서 약 `0.92~0.96 sec/iter`, epoch당 약 `61,790 iter`가 나왔습니다.

- baseline full train: 약 4.5~5일
- reliability full train: 약 5일 안팎
- baseline + reliability 순차 실행: 약 9~10일

장비, 스토리지 I/O, validation 시간에 따라 달라질 수 있습니다.

## 현재 해석

이 모델은 clean mAP를 자동으로 올리는 구조라기보다, LiDAR base를 유지하면서 camera/LiDAR correction을 reliability gate로 조절하는 robustness-oriented 구조입니다. 따라서 논문/보고서에서는 clean 성능과 corruption 성능을 같이 비교하는 것이 좋습니다.

권장 실험 순서:

1. baseline clean validation
2. reliability clean validation
3. gate collapse 여부 확인
4. image/LiDAR corruption matrix 평가
5. ablation 비교: residual only, image gate, LiDAR gate, image+LiDAR gate

## 주의

- GitHub에는 데이터와 체크포인트를 올리지 마세요.
- `NUSCENES_DATA_ROOT`, `BEVFUSION_WORK_ROOT`는 각자 환경에 맞게 지정하세요.
- 오래 걸리는 full train 전에 반드시 100iter debug run과 dataset sample check를 먼저 통과시키세요.
