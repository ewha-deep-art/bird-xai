# Bird XAI — 구현 가이드

프로젝트 개요와 디렉토리 설명은 [README.md](README.md) 및 각 폴더의 `README.md`를 참고합니다.

## 구현 원칙

- 구현은 `training -> inference -> server -> render` 흐름을 기준으로 진행합니다.
- 각 phase는 독립적으로 개발 가능해야 하지만, 최종적으로는 하나의 실시간 파이프라인으로 연결되어야 합니다.
- Python과 Unity의 병렬 작업은 `contracts/`의 스키마를 기준으로 맞춥니다.

## 구현 순서

### 0. Contracts 초안 및 Mock 서버

**목표**: 실제 ML 파이프라인 없이도 Unity 연동 개발을 시작할 수 있는 상태

- `contracts/schemas/` — 서버↔Unity 간 메시지 스키마 확정
- `ai/contracts/models.py` — Pydantic 모델
- `contracts/examples/frame.sample.json` — 픽스처 데이터
- `ai/server/` — WebSocket 서버 (`BIRD_XAI_MOCK=true`로 픽스처 주기 전송)

### 1. 데이터 준비

**목표**: 학습에 사용할 정제된 데이터셋 생성

- `data/processed/` — GPS + ERA5 전처리 결과

### 2. Training phase

**목표**: 경로 예측 모델 학습 및 가중치 저장

- `ai/training/model/weights/` — 추론에 바로 사용할 수 있는 가중치

### 3. Inference phase

**목표**: 가중치를 사용해 경로·XAI·boids 결과를 생성하는 모듈

- `ai/inference/predictor/` — 대표 경로 + 후보 경로 반환
- `ai/inference/xai/` — feature별 기여도 (정규화 포함)
- `ai/inference/boids/` — 군집 시뮬레이션 (위치 + 속도)

### 4. Contracts 갱신

**목표**: inference 실제 출력에 맞춰 0단계 스키마 검증 및 수정

- `attributionFeatureKey` enum을 실제 학습 feature로 교체
- `contracts/examples/frame.sample.json`을 실제 데이터로 교체
- 스키마·Pydantic 모델 불일치 해소

### 5. Server phase

**목표**: inference 결과를 Unity에 주기적으로 전달하는 서버 완성

- `ai/inference/pipeline.py` — predictor + xai + boids를 조합하는 진입점
- `ai/server/service.py` `iter_frames()` — 윈도우 큐 기반 프레임 전송

### 6. Render 연동

**목표**: Unity에서 경로·XAI·군집을 실시간으로 시각화하고 `controls.set` 루프 연결

## 결정이 필요한 항목

- 입력 윈도우 길이와 출력 길이 (현재 코드: observed 48 steps, predicted 12 steps, 15분 간격)
- 조작 가능 feature 목록 (현재 코드: `wind_speed`, `wind_direction`)
- boids 시뮬레이션의 개체 수와 업데이트 주기
- Unity를 현재 레포에 유지할지, 이후 별도 레포로 분리할지

## 확정된 항목

- 후보 경로: 3개
- XAI: 윈도우당 단일 attribution (구간 분리 없음), 서버에서 0-1 정규화

## 통합 기준

- `training` 산출물은 `inference`에서 재가공 없이 사용할 수 있어야 합니다.
- `inference` 산출물은 `server`에서 계약 문서 기준으로 조합 가능해야 합니다.
- `server` payload는 Unity mock 없이도 검증 가능해야 합니다.
- `render`는 실제 모델 연결 전에도 `contracts/` 스키마 기준 mock 데이터로 개발 가능해야 합니다.
