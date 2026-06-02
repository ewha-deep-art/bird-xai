# Bird XAI

![idea preview](docs/idea_v2.png)

Bird XAI는 철새 GPS·기후 데이터로 비행 경로를 예측하고, XAI로 판단 근거를 시각화하는 인터랙티브 미디어아트 프로젝트입니다. 관람객은 QR 웹에서 메시지를 보내면 AI 예측 입력이 바뀌고, Unity 화면에 경로·기여도가 실시간으로 반영됩니다.

예측 경로, XAI attribution, Unity 군집(Boids), 실시간 렌더링이 하나의 파이프라인으로 연결됩니다.

## 프로젝트 흐름

1. **학습** — 전처리 데이터로 LSTM 경로 예측 모델 학습
2. **추론** — Captum IG XAI + queue 기반 frame 생성
3. **전달·렌더링** — FastAPI `/ws` → Unity, 관람 `/wish` → override

```text
Movebank GPS + ERA5
  -> data/processed
  -> Training
  -> Inference (BirdPipeline)
  -> Server (/ws frame, /wish)
  -> Unity Render
```

## 개요

- 데이터: `Movebank`, `ERA5`
- AI: Python (`PyTorch`, `captum`)
- 렌더: `Unity` (`render/`, monorepo)

## 디렉토리 안내

| 경로 | 설명 |
|---|---|
| [docs/README.md](docs/README.md) | 기획 문서 목록 |
| [docs/unimplemented.md](docs/unimplemented.md) | 미구현·보류 레지스트리 (A/B/C/D) |
| [docs/deploy.md](docs/deploy.md) | Railway·Docker·artifacts·CI runbook |
| [data/README.md](data/README.md) | raw / processed 데이터 |
| [ai/README.md](ai/README.md) | AI 파이프라인 |
| [ai/training/README.md](ai/training/README.md) | 학습 |
| [ai/inference/README.md](ai/inference/README.md) | 추론 |
| [ai/server/README.md](ai/server/README.md) | FastAPI 서버 |
| [contracts/README.md](contracts/README.md) | Unity 계약 (JSON Schema) |
| [render/README.md](render/README.md) | Unity 렌더 |
| [tests/README.md](tests/README.md) | WebSocket smoke test |

## 참고

- 데이터·모델 가중치는 용량상 Git에 포함하지 않을 수 있음
- 구현 스냅샷: [AGENTS.md](AGENTS.md), [docs/unimplemented.md](docs/unimplemented.md)

## 환경 설정

```bash
uv venv
uv sync --extra dev
```

CLI (`pyproject.toml` `[project.scripts]`):

- `bird-xai-preprocess` — 전처리 (Python 포팅 예정, entry만 등록)
- `bird-xai-train` — 학습
- `bird-xai-server` — 서버
- `bird-xai-ws-smoke` — WebSocket smoke test

## 실행 순서

`data/processed/`와 `model/weights/bird_best.pt`가 준비된 경우:

```bash
uv run bird-xai-train --epochs 1   # 가중치 재생성 시
uv run bird-xai-server
```

전처리 CLI(`bird-xai-preprocess`)는 [docs/unimplemented.md](docs/unimplemented.md) B항 — 노트북→Python 포팅 후 사용.

## 빠른 검증

```bash
uv run bird-xai-server &
uv run bird-xai-ws-smoke
# QR 관람 웹: http://127.0.0.1:8080/wind-and-wish
```

배포 runbook → [docs/deploy.md](docs/deploy.md)
