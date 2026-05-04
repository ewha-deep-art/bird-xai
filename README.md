# Bird XAI

Bird XAI는 철새의 GPS 이동 데이터와 기후 데이터를 함께 분석해 미래 비행 경로를 예측하고, 그 판단 근거를 XAI로 시각화하는 인터랙티브 미디어아트 프로젝트입니다. 관객은 환경 변수를 조작하면서 AI의 결과가 어떻게 달라지는지 실시간으로 경험하게 됩니다.

이 프로젝트는 하나의 정답 경로를 제시하기보다, 자연 현상을 데이터와 모델로 해석하는 과정이 얼마나 가변적이고 불확실한지를 드러내는 데 초점을 둡니다. 예측 경로, 후보 경로, 설명 값, 군집 움직임, 실시간 렌더링이 하나의 파이프라인으로 연결되어 작품의 핵심 경험을 만듭니다.

![idea preview](docs/idea_v2.png)

## 프로젝트 흐름

프로젝트는 크게 세 개의 phase로 구성됩니다.

1. 학습
   GPS 및 기후 데이터를 정제하고 모델 학습에 사용할 입력을 준비합니다.
2. 추론
   입력 조건에 따라 경로를 예측하고, XAI와 군집 시뮬레이션을 적용합니다.
3. 전달 및 렌더링
   결과를 Unity로 전달하고 실시간으로 시각화합니다.

```text
Movebank GPS + ERA5 Climate
  -> Training
  -> Inference
  -> Server
  -> Unity Render
```

## 개요

- 데이터 출처: `Movebank`, `ERA5`
- 학습 및 추론: Python 기반 AI 파이프라인
- 렌더링: `Unity`

## 디렉토리 안내

| 경로 | 설명 |
|---|---|
| [docs/README.md](docs/README.md) | 작품 기획, 시각적 레퍼런스, 개념 문서 |
| [data/README.md](data/README.md) | 원본 데이터와 전처리 산출물 저장 구조 |
| [ai/README.md](ai/README.md) | AI 파이프라인 전체 구조 |
| [ai/training/README.md](ai/training/README.md) | 학습 phase 설명 |
| [ai/inference/README.md](ai/inference/README.md) | 추론 phase 설명 |
| [ai/server/README.md](ai/server/README.md) | Unity 전달을 위한 서버 레이어 설명 |
| [contracts/README.md](contracts/README.md) | Python과 Unity 사이 계약 문서 설명 |
| [render/README.md](render/README.md) | Unity 렌더링과 인터랙션 레이어 설명 |

## 참고

- 일부 데이터와 모델 가중치는 용량 문제로 Git에 포함하지 않습니다.
