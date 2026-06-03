# ai/

Python AI 파이프라인: 학습 → 추론 → FastAPI 서버. Unity는 `render/`의 `BirdDataManager`가 `WS /ws`로 frame을 수신합니다.

공개 체험 URL·관람 규칙(10회 flush 등)은 루트 [README.md](../README.md) · [self_demo.md](../self_demo.md).

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
