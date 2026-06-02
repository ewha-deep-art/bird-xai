# ai/

이 디렉토리는 프로젝트의 Python 기반 AI 파이프라인을 담습니다. 학습, 추론, Unity 전달용 서버가 이 안에서 단계적으로 연결됩니다.

## 구성

- `training/`
  - 모델 학습 및 가중치 저장
- `inference/`
  - 학습된 모델을 사용한 경로 예측 및 Captum IG XAI
- `server/`
  - 추론 결과를 Unity에 전달하는 서버 레이어
- `common/`
  - 공통 상수 및 경로
  - Pydantic 계약 모델 (`models.py`)
- `config.py`
  - 서버 설정 (`BIRD_XAI_*` 환경변수)

## 주요 기술

- `pandas`, `numpy`, `scipy`, `xarray`
- `PyTorch`, `captum`
- `FastAPI`, `Pydantic`, `WebSocket`

## 관련 문서

- [training/README.md](training/README.md)
- [inference/README.md](inference/README.md)
- [server/README.md](server/README.md)
