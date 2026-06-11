# 동일 Night 데이터 비교 결과

동일한 night 프레임 하나에 공식 BEVFusion 체크포인트의 동일한 예측을
사용하고, 표시 confidence threshold만 변경한 결과다.

- scene: `scene-1073`
- token: `a669c657a1f54e6a9a776003aeb66a72`
- 상황: Night, scooters, turn left, dense traffic

## 잘 인식한 결과

- 결과: [`same_sample_good_bev.png`](same_sample_good_bev.png)
- threshold: 0.1
- TP / FP / FN: 14 / 0 / 0
- precision / recall / F1: 1.000 / 1.000 / 1.000

## 객체 3개를 빠뜨린 결과

- 결과: [`same_sample_missed3_bev.png`](same_sample_missed3_bev.png)
- threshold: 0.5
- TP / FP / FN: 11 / 0 / 3
- precision / recall / F1: 1.000 / 0.786 / 0.880

오른쪽 BEV에서 빨간색 박스 3개가 누락된 GT다. 임의로 박스를 삭제한 것이
아니라 confidence가 0.5보다 낮은 예측 3개가 후처리에서 제외된 결과다.
