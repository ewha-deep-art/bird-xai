# model/ — AI 모델

## 구조

```
model/
├── predictor/     경로 예측 모델 (LSTM)
├── boids/         군집 시뮬레이션 (Boids 알고리즘)
├── xai/           설명 가능한 AI (SHAP 기여도 계산)
├── weights/       최종 모델 가중치 (git 제외)
└── checkpoints/   학습 중간 체크포인트 (git 제외)
```

## predictor/ — 경로 예측

철새의 현재 경로와 환경 데이터를 입력받아 미래 경로를 예측합니다.

## boids/ — 군집 시뮬레이션

예측된 GPS 경로를 리더 개체로 삼아 가상 파티클 군집을 시뮬레이션합니다.

## xai/ — 설명 가능한 AI

SHAP으로 각 환경변수가 경로 예측에 얼마나 기여했는지 수치화합니다.

## 가중치 파일

학습된 가중치(`.pt`, `.ckpt`)는 git에서 제외됩니다.

```bash
python model/predictor/train.py
# → model/weights/ 에 저장됨
```
