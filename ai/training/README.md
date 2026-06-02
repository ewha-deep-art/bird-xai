# ai/training/

이 디렉토리는 프로젝트의 학습 phase를 담당합니다. 전처리된 데이터를 입력받아 **미래 경로 예측** 모델을 학습시키며, 체크포인트와 최종 가중치를 관리합니다.

## 주요 역할

- Direct multi-step forecast 모델 정의 및 로드
- 과거 24 step → 미래 12 step Δ target 학습
- 체크포인트와 최종 가중치 관리

## 내부 구성

- `model/`
  - `__init__.py` — 모델 구조 정의 및 로드 유틸리티
    - `BirdForecastLSTM` — LSTM encoder + direct future head
      - 입력: `(batch, 24, input_size)` — 과거 GPS·운동·기상
      - 출력: `(batch, 12, 3)` — 미래 Δlat, Δlon, Δheight
    - `load_model()` — 기본 하이퍼파라미터로 `BirdForecastLSTM` 인스턴스 생성
    - `load_model_with_state()` — 저장된 가중치를 로드하여 추론 준비 완료 상태로 반환
  - `weights/` — 최종 모델 가중치 (`bird_best.pt`)
- `train.py`
  - 고정된 최적 하이퍼파라미터로 단일 학습 실행
  - val **lat RMSE (degree)** 기준으로 `model/weights/bird_best.pt` 저장
  - epoch마다 persistence baseline lat RMSE와 비교 출력
  - test: Δ 역변환 → 절대 좌표 RMSE, horizon별 lat RMSE, persistence 대비 여부
- `experiment.py`
  - Optuna를 사용한 하이퍼파라미터 자동 탐색

## 사용 기술

- `PyTorch`, `optuna`, `scikit-learn` (Δ scaler는 `data/loader.py`에서 fit)

## 워크플로우

```bash
# 하이퍼파라미터 탐색 (선택)
uv run python -m ai.training.experiment --n_trials 30 --epochs 50
# → ./checkpoints/bird_best.pt (experiment 전용)

# 확정 파라미터·기본 LR로 학습
uv run bird-xai-train
# → ai/training/model/weights/bird_best.pt

# 추론·서버 로드
from ai.training.model import load_model_with_state
model = load_model_with_state()
```

## 입력과 출력

- 입력:
  - `ai.common`의 data loader (`train_loader`, `val_loader`) — `(X, y_delta, last_obs)`
- 출력:
  - `checkpoints/bird_best.pt` — `experiment.py` Optuna 재학습용 (repo 루트 cwd 기준)
  - `ai/training/model/weights/bird_best.pt` — `train.py` 최종 가중치 (`MODEL_SAVE_PATH`)
  - `data/processed/*.pkl` — `loader.py` normalize 시 갱신 (import `ai.common` 시에도 fit)

데이터 전처리와 데이터로더 관련 내용은 [data/README.md](../../data/README.md), 추론 관련 내용은 [../inference/README.md](../inference/README.md)를 참고하세요.
