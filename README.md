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

## 환경 설정

현재 저장소는 `pyproject.toml`의 `[project.scripts]`로 CLI를 등록합니다. 따라서 `bird-xai-preprocess`, `bird-xai-train`, `bird-xai-server`를 쓰려면 먼저 프로젝트 환경을 설치해야 합니다.

`uv` 기준 권장 설치 순서:

```bash
cd /home/nagyeop/bird-xai
uv venv
uv sync --extra dev
```

가상환경을 활성화해서 콘솔 스크립트를 직접 쓰려면:

```bash
source .venv/bin/activate
bird-xai-preprocess --help
```

가상환경을 활성화하지 않고 바로 실행하려면 `uv run`을 사용합니다.

```bash
uv run bird-xai-preprocess --help
```

현재 기본 raw 데이터 경로는 아래 두 파일로 고정되어 있습니다.

- `data/raw/H17-6330-6330.csv`
- `data/raw/1f235b2421969a15a264a061fe577e4b.nc`

## 최소 기능 실행 순서

실행 순서는 아래 하나로 고정합니다.

```bash
uv run bird-xai-preprocess
uv run bird-xai-train --epochs 1
uv run bird-xai-server
```

`bird-xai-preprocess`는 항상 strict ERA5 매핑을 사용합니다. 따라서 `xarray`, `netCDF4` 같은 의존성과 실제 ERA5 변수 매핑이 모두 준비되어 있어야 합니다.

## 빠른 검증

mock 서버를 띄운 상태에서 WebSocket 왕복 검증:

```bash
BIRD_XAI_MOCK=true uv run bird-xai-server &
uv run bird-xai-ws-smoke
```
