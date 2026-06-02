# Bird XAI

![idea preview](docs/idea_v2.png)

Bird XAI는 철새 GPS·기후 데이터로 비행 경로를 예측하고, Captum IG로 판단 근거를 시각화하는 인터랙티브 미디어아트입니다. 관람객은 QR 웹(`/wind-and-wish`)에서 메시지를내면 예측 입력(`ws_850`)이 바뀌고, Unity는 `/ws`로 경로·XAI를 실시간 수신합니다.

## 프로젝트 흐름

1. **학습** — `data/processed` + LSTM direct multi-step forecast
2. **추론** — Captum IG + queue 기반 `FrameMessage`
3. **전달** — FastAPI `/ws` → Unity, `/wish` → override

```text
Movebank GPS + ERA5
  → data/processed
  → Training (BirdForecastLSTM)
  → Inference (BirdPipeline)
  → Server (/ws frame, /wish)
  → Unity (render/, B항)
```

## 스택

- 데이터: Movebank, ERA5
- AI: Python (`PyTorch`, `captum`)
- 서버: FastAPI, WebSocket
- 렌더: Unity (`render/`, monorepo — C#/WebGL B항)

## 디렉토리

| 경로 | 설명 |
|---|---|
| [docs/unimplemented.md](docs/unimplemented.md) | 미구현·보류 (A/B/C/D) |
| [docs/deploy.md](docs/deploy.md) | Railway·Docker·CI |
| [data/README.md](data/README.md) | loader, processed 산출물 |
| [ai/README.md](ai/README.md) | 학습·추론·서버 |
| [contracts/README.md](contracts/README.md) | Unity **WS** JSON Schema |
| [render/README.md](render/README.md) | Unity |
| [tests/README.md](tests/README.md) | `bird-xai-ws-smoke` |

## 환경

```bash
uv sync --extra dev
```

CLI (`pyproject.toml`):

| 명령 | 설명 |
|---|---|
| `bird-xai-train` | 학습 → `ai/training/model/weights/bird_best.pt` |
| `bird-xai-server` | API + 관람 웹 + `/ws` |
| `bird-xai-ws-smoke` | frame·wish smoke test |

## 실행

`data/processed/` 3파일 + `bird_best.pt`가 repo에 있으면:

```bash
uv run bird-xai-train --epochs 1   # 가중치 재학습 시
uv run bird-xai-server
```

## 빠른 검증

```bash
BIRD_XAI_WISH_RATE_LIMIT_SEC=0 uv run bird-xai-server &
uv run bird-xai-ws-smoke --wish-interval 0
```

- 관람 웹: http://127.0.0.1:8080/wind-and-wish
- 배포: [docs/deploy.md](docs/deploy.md)

## 참고

- 구현 규칙·고정값: [AGENTS.md](AGENTS.md)
- 일부 대용량 raw는 Git 미포함
