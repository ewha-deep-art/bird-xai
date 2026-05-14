# Bird XAI — 구현 가이드

프로젝트 개요와 디렉토리 설명은 [README.md](README.md) 및 각 폴더의 `README.md`를 참고합니다. 이 문서는 현재 저장소 구조를 기준으로, 실제 구현을 어떤 순서와 기준으로 진행할지 정리하는 작업 문서입니다.

## 구현 원칙

- 구현은 `training -> inference -> server -> render` 흐름을 기준으로 진행합니다.
- 각 phase는 독립적으로 개발 가능해야 하지만, 최종적으로는 하나의 실시간 파이프라인으로 연결되어야 합니다.
- Python과 Unity의 병렬 작업은 `contracts/`의 스키마를 기준으로 맞춥니다.
- README에는 폴더 설명을 두고, AGENTS에는 구현 순서, 결정 기준, 통합 계획을 둡니다.

## 현재 기준 디렉토리

```text
data/
  raw/
  processed/

ai/
  training/
    preprocessing/
    model/
  inference/
    predictor/
    xai/
    boids/
  server/

contracts/
render/
```

## 구현 순서

### 1. 데이터 준비

- `data/raw/`에 Movebank GPS CSV와 ERA5 원본 파일을 수집합니다.
- 학습에 사용할 개체, 시간 범위, 공간 범위를 먼저 고정합니다.
- Movebank와 ERA5에서 실제로 어떤 필드를 사용할지 정리하고, 이후 스키마에 반영합니다.

### 2. Training phase 구현

`ai/training/`은 학습용 데이터를 만들고 모델을 학습시키는 단계입니다.

#### 2-1. preprocessing

- GPS 시계열을 개체별, 날짜별로 정렬합니다.
- 실제 비행 구간을 분리합니다.
- 이상치 제거 규칙을 정합니다.
- Kalman 필터 또는 유사 방식으로 GPS 노이즈를 줄입니다.
- spline 보간으로 시간 간격을 정규화합니다.
- ERA5 기후 변수를 GPS 경로의 시간, 위치, 고도 기준으로 매핑합니다.
- 산출물은 `data/processed/`에 저장합니다.

#### 2-2. model

- 전처리 결과를 입력으로 사용하는 경로 예측 모델을 구현합니다.
- 1차 목표는 대표 경로 1개와 후보 경로 여러 개를 반환할 수 있는 형태를 만드는 것입니다.
- 불확실성 표현은 Monte Carlo Dropout 또는 이에 준하는 샘플링 전략을 우선 검토합니다.
- 학습 결과는 `ai/training/model/checkpoints/`와 `ai/training/model/weights/`에 나눠 저장합니다.

### 3. Inference phase 구현

`ai/inference/`는 학습된 가중치를 사용해 실시간 또는 반실시간 결과를 생성하는 단계입니다.

#### 3-1. predictor

- `ai/training/model/weights/`의 가중치를 불러옵니다.
- 입력 경로와 환경 변수를 받아 대표 경로와 후보 경로를 생성합니다.
- 출력 형식은 server와 Unity가 바로 사용할 수 있도록 일정해야 합니다.

#### 3-2. xai

- predictor 결과와 입력 feature를 기반으로 기여도를 계산합니다.
- 초기 버전은 SHAP 기반으로 설계합니다.
- 출력은 feature별 기여도, 시점별 또는 구간별 설명 값으로 정리합니다.
- 값 범위는 Unity에서 시각화하기 쉬운 형태로 정규화 가능한 구조여야 합니다.

#### 3-3. boids

- 대표 경로를 리더 경로로 사용합니다.
- Separation, Alignment, Cohesion 규칙을 적용한 군집 시뮬레이션을 계산합니다.
- 출력은 프레임별 위치와 속도처럼 렌더링에 바로 넘길 수 있는 형태를 우선합니다.

### 4. Contracts 정리

`contracts/`는 Python과 Unity가 병렬 작업할 수 있도록 먼저 정의합니다.

초기 우선순위는 아래와 같습니다.

1. `frame-schema`
   - 대표 경로
   - 후보 경로
   - XAI 결과
   - boids 결과
   - 현재 환경 변수 상태
2. `interaction-schema`
   - Unity에서 Python으로 보내는 환경 변수 조작 입력
   - 타임라인 이동, 특정 개체 선택 등 상호작용 이벤트
3. `model-output-schema`
   - predictor 출력 형식
4. `xai-schema`
   - feature 이름, 기여도 값, 시간축 대응 방식

구현 중에는 Python 쪽 Pydantic 모델과 Unity 쪽 C# 타입이 이 계약을 같이 따르도록 유지합니다.

### 5. Server phase 구현

`ai/server/`는 inference 결과를 Unity에 전달하는 레이어입니다.

- FastAPI와 WebSocket 기반으로 구성합니다.
- 최소 전달 단위는 프레임 또는 시점 단위 JSON payload입니다.
- predictor, xai, boids 출력을 하나의 응답 형식으로 조합합니다.
- Unity에서 입력이 오면 필요한 재계산 경로를 다시 실행합니다.
- 초기에는 단일 클라이언트 기준으로 구현하고, 이후 다중 세션이 필요하면 확장합니다.

### 6. Render 연동

`render/`는 Unity 프로젝트 또는 Unity 관련 문서가 위치하는 영역입니다.

- server에서 정의한 payload를 기준으로 경로, 설명 값, 군집을 시각화합니다.
- 초기 목표는 아래 세 가지가 동시에 보이는 장면입니다.
  - 대표 경로
  - 후보 경로
  - XAI 또는 군집 기반 시각적 변화
- 이후 인터랙션 입력을 server에 되돌려 보내는 루프를 연결합니다.

## 단계별 산출물

### Milestone 1

- `data/raw/` 샘플 데이터 확보
- 전처리 파이프라인 초안
- `data/processed/` 샘플 산출물 생성

### Milestone 2

- 학습 가능한 모델 입력 구성 완료
- 경로 예측 모델 학습 코드 초안
- 체크포인트 저장 및 재로딩 가능 상태

### Milestone 3

- predictor가 대표 경로와 후보 경로를 반환
- xai가 feature 기여도 값을 반환
- boids가 기본 군집 상태를 반환

### Milestone 4

- `contracts/`의 핵심 스키마 초안 완료
- `ai/server/`에서 통합 payload 송신 가능
- Unity 또는 mock client에서 payload 수신 가능

### Milestone 5

- Unity에서 경로, 설명 값, 군집 상태 동시 시각화
- 환경 변수 조작 입력이 Python 파이프라인 재실행으로 연결

## 결정이 필요한 항목

- 모델의 입력 윈도우 길이와 출력 길이
- 후보 경로 개수
- XAI를 프레임 단위로 낼지, 구간 단위로 낼지
- boids 시뮬레이션의 개체 수와 업데이트 주기
- Unity를 현재 레포에 유지할지, 이후 별도 레포로 분리할지

## 통합 기준

- `training` 산출물은 `inference`에서 재가공 없이 사용할 수 있어야 합니다.
- `inference` 산출물은 `server`에서 계약 문서 기준으로 조합 가능해야 합니다.
- `server` payload는 Unity mock 없이도 검증 가능해야 합니다.
- `render`는 실제 모델 연결 전에도 `contracts/` 스키마 기준 mock 데이터로 개발 가능해야 합니다.
