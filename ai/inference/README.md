# ai/inference/

이 디렉토리는 프로젝트의 추론 phase를 담당합니다. 학습된 모델과 입력 조건을 바탕으로 경로를 예측하고, 그 결과를 설명 값과 군집 시뮬레이션으로 확장합니다.

## 주요 역할

- 학습된 가중치 로드
- 입력 조건에 따른 대표 경로와 후보 경로 생성
- feature 기여도 계산
- 시각화를 위한 군집 상태 생성

## 내부 구성

- `predictor/`
  - 학습된 모델을 이용한 경로 예측
- `xai/`
  - 예측 결과에 대한 feature 기여도 계산
- `boids/`
  - 예측 경로 기반 군집 시뮬레이션

## 사용할 기술

- `PyTorch`
- `shap`
- `numpy`

## 입력과 출력

- 입력:
  - `../training/model/weights/`의 학습된 모델
  - 전처리된 데이터 또는 실시간 환경 입력
- 출력:
  - 대표 경로
  - 후보 경로 집합
  - feature 기여도
  - 군집 시뮬레이션 상태

관련 전달 레이어는 [../server/README.md](../server/README.md), 계약 문서는 [../../contracts/README.md](../../contracts/README.md)를 참고하세요.
