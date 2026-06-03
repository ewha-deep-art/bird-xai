# tests/

Bird XAI 서버 **WebSocket frame** + **`POST /wish` override** smoke test (`bird-xai-ws-smoke`).

## 구성

- `smoke_client.py` — 터미널 smoke (`tests.smoke_client:main`)

## 실행

```bash
BIRD_XAI_WISH_RATE_LIMIT_SEC=0 uv run bird-xai-server &
uv run bird-xai-ws-smoke --wish-interval 0
```

기본 순서:

1. `GET /wind-and-wish` HTML 검증
2. (선택) `--with-negative` — `/wish` 400·429
3. `WS /ws` → bootstrap `frame`
4. `POST /wish` × flush threshold (기본 10)
5. stream에서 `applied_overrides.message_cnt` 확인
6. 추가 frame (기본 5)

## 옵션

```bash
uv run bird-xai-ws-smoke --no-with-wish --frames 10
uv run bird-xai-ws-smoke --wish-interval 5.1 --flush-threshold 10
uv run bird-xai-ws-smoke --with-negative --wish-interval 0
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--base-url` | `http://127.0.0.1:8080` | `/wish`, `/health` |
| `--url` | (derived) | WS URL override |
| `--timeout` | `600` | step timeout (초) |
| `--frames` | `5` | wish 후 추가 frame 수 |
| `--with-wish` / `--no-with-wish` | on | override 시나리오 |
| `--with-negative` | off | 400/429 |
| `--flush-threshold` | `10` | `BIRD_XAI_WISH_FLUSH_THRESHOLD` |
| `--wish-interval` | `0` | wish POST 간격 (CI: `0`) |

## CI

GHA: [`.github/workflows/smoke.yml`](../.github/workflows/smoke.yml) — `BIRD_XAI_WISH_RATE_LIMIT_SEC=0`. artifacts 4종·cache seed → [docs/deploy.md](../docs/deploy.md) · [docs/unimplemented.md](../docs/unimplemented.md) C항.

## 관련

- HTTP·WS API: [ai/server/README.md](../ai/server/README.md)
- Unity frame 계약: [contracts/README.md](../contracts/README.md)
