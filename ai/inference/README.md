# ai/inference/

이 디렉토리는 프로젝트의 실시간 추론(inference) 파이프라인을 담당합니다.  
학습된 모델을 기반으로 새의 위치를 예측하고, XAI(설명 가능 AI) 결과와 함께 프레임 단위 데이터로 가공하여 전달합니다.

현재 구현은 `BirdPipeline` 중심의 단일 파이프라인 구조로 구성되어 있으며, 예측 결과를 queue 기반으로 관리합니다.

---

## 주요 역할

- 학습된 모델 및 scaler 로드
- 입력 시퀀스 기반 위치 예측
- Integrated Gradients 기반 feature attribution 계산
- 프레임 단위 메시지 생성
- predicted path queue 관리
- override 입력 반영 준비 구조 제공

---

## 내부 구성

- `pipeline.py`
  - 추론 전체 흐름 관리
  - queue 기반 frame 생성
  - prediction + XAI orchestration 수행

---

## 현재 파이프라인 구조

```text
test_loader
    ↓
BirdPipeline._build_queue()
    ├── predict()
    │     └── 위치(lat, lon, altitude) 예측
    │
    └── apply_xai()
          └── Integrated Gradients attribution 계산
    ↓
(queue 저장)
    ↓
build_frame()
    ↓
FrameMessage 반환
```

## 관련 디렉토리
- 서버 전달 레이어: [ai/server/README.md](../server/README.md)
- 계약 및 스키마: [contracts/README.md](../../contracts/README.md)