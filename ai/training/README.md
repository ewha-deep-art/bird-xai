# ai/training/

이 디렉토리는 프로젝트의 학습 phase를 담당합니다. 원본 GPS 데이터와 기후 데이터를 정제해 학습 가능한 입력으로 바꾸고, 경로 예측 모델을 학습시키며, 체크포인트와 최종 가중치를 관리합니다.

## 주요 역할

- GPS 및 기후 데이터 전처리
- 학습용 데이터셋 생성
- 경로 예측 모델 학습
- 체크포인트와 최종 가중치 관리

## 내부 구성

- `preprocessing/`
  - GPS 및 기후 데이터 전처리
- `model/`
  - 학습 코드, 체크포인트, 최종 가중치 관리

## 사용할 기술

- 데이터 처리:
  - `pandas`
  - `numpy`
  - `xarray`
- 수치 처리 및 전처리:
  - `scipy`
  - `filterpy`
- 딥러닝:
  - `PyTorch`

## 입력과 출력

- 입력:
  - `data/raw/`의 Movebank CSV
  - `data/raw/`의 ERA5 NetCDF
- 출력:
  - `data/processed/`의 학습용 데이터
  - `model/checkpoints/`의 중간 학습 상태
  - `model/weights/`의 최종 모델 가중치

관련 데이터 저장 위치는 [data/README.md](../../data/README.md), 추론 phase는 [../inference/README.md](../inference/README.md)를 참고하세요.
