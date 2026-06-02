# ai/

Python AI 파이프라인: 학습 → 추론 → FastAPI 서버. Unity는 `render/`에서 WebSocket으로 frame을 수신합니다.

## 흐름

```text
data/processed + model/weights/bird_best.pt
  → training/     (학습·가중치)
  → inference/    (BirdPipeline: predict + Captum IG + queue)
  → server/       (/ws frame, /wish override, /wind-and-wish)
```

## 구성

| 경로 | 역할 |
|---|---|
| `common/` | 상수·경로 (`__init__.py`), Pydantic 메시지 (`models.py`) |
| `config.py` | 서버 env (`BIRD_XAI_*`) |
| `training/` | `BirdForecastLSTM`, `train.py`, `experiment.py`, `model/weights/` |
| `inference/` | `BirdPipeline` (`pipeline.py`) |
| `server/` | FastAPI, WebSocket, 관람 HTTP |

## 메시지·계약

- `models.py`: `FrameMessage`, `ErrorMessage`, `WishRequest`, `WishResponse` 등
- Unity `WS /ws` JSON Schema: [contracts/](../contracts/)

## CLI

```bash
uv run bird-xai-train          # 학습
uv run bird-xai-server         # 서버
uv run bird-xai-ws-smoke       # smoke test
```

## 관련

- [training/README.md](training/README.md)
- [inference/README.md](inference/README.md)
- [server/README.md](server/README.md)
- [AGENTS.md](../AGENTS.md) — 구현 스냅샷
