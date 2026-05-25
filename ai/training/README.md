# ai/training/

이 디렉토리는 프로젝트의 학습 phase를 담당합니다. 전처리된 데이터를 입력받아 경로 예측 모델을 학습시키며, 체크포인트와 최종 가중치를 관리합니다.

## 주요 역할

- 경로 예측 모델 정의 및 로드
- 경로 예측 모델 학습 (단순 학습 / 하이퍼파라미터 탐색)
- 체크포인트와 최종 가중치 관리

## 내부 구성

- `model/`
  - `__init__.py` — 모델 구조 정의 및 로드 유틸리티
    - `BirdLSTM` — 단층/다층 LSTM + FC 출력 모델
      - `input_size`: 입력 피처 수
      - `hidden_size`: LSTM 은닉 상태 크기
      - `num_layers`: LSTM 레이어 수 (2층 이상일 때만 dropout 적용)
      - `output_size`: 출력 타겟 수
      - `dropout`: 드롭아웃 비율
      - 출력 shape: `(batch, window_size, output_size)`
    - `load_model()` — 기본 하이퍼파라미터로 `BirdLSTM` 인스턴스 생성
    - `load_model_with_state()` — 저장된 가중치를 로드하여 추론 준비 완료 상태로 반환
  - `weights/` — 최종 모델 가중치 저장 (예: `Art_best.pt`)
- `train.py`
  - 고정된 최적 하이퍼파라미터로 단일 학습 실행
  - val loss 기준으로 `model/weights/bird_best.pt` 저장
- `experiment.py`
  - Optuna를 사용한 하이퍼파라미터 자동 탐색

## 사용 기술

- 딥러닝: `PyTorch`
- 하이퍼파라미터 최적화: `optuna`

## 워크플로우

```
[처음 실험 / 하이퍼파라미터 탐색]
uv run python experiment.py --bird Art --n_trials 30 --epochs 50

      ↓  탐색 완료 후 best params 출력

[확정된 파라미터로 학습]
uv run bird-xai-train

      ↓  model/weights/bird_best.pt 저장

[학습 완료된 모델 로드]
from ai.training.model import load_model_with_state
model = load_model_with_state(bird="Art")  # eval 모드로 반환
```

## 입력과 출력

- 입력:
  - `ai.common`의 data loader (`train_loader`, `val_loader`)
- 출력:
  - `checkpoints/bird_best.pt` — experiment.py의 중간 체크포인트
  - `model/weights/bird_best.pt` — train.py 및 experiment.py의 최종 모델 가중치

데이터 전처리와 데이터로더 관련 내용은 [data/README.md](../../data/README.md), 추론 관련 내용은 [../inference/README.md](../inference/README.md)를 참고하세요.