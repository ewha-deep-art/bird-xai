# tests/

Bird XAI 서버 WebSocket frame 스트리밍 + `/wish` override smoke test입니다.

## 주요 역할

- WebSocket 연결 및 bootstrap frame 검증
- Pydantic schema validation
- `GET /wind-and-wish` HTML 페이지 검증
- `POST /wish` flush → `applied_overrides.message_cnt` 확인
- (선택) 400/429 negative checks

## 내부 구성

- `smoke_client.py` — 터미널 smoke test (`bird-xai-ws-smoke`)

## Smoke test

```bash
uv run bird-xai-server &
uv run bird-xai-ws-smoke
```

기본 순서:

1. `GET /wind-and-wish` HTML 검증
2. (선택) negative `/wish` checks (`--with-negative`)
3. WS connect → bootstrap frame
4. `POST /wish` × flush threshold (default 10)
5. frame stream에서 `applied_overrides.message_cnt` 확인
6. 추가 frame stream (default 5 frames)

### 옵션

```bash
# 로컬 (rate limit 회피 — CI와 동일)
BIRD_XAI_WISH_RATE_LIMIT_SEC=0 uv run bird-xai-server &
uv run bird-xai-ws-smoke --wish-interval 0

# production-like rate limit
uv run bird-xai-ws-smoke --wish-interval 5.1 --flush-threshold 10

# WS only (wish 생략)
uv run bird-xai-ws-smoke --no-with-wish --frames 10

# negative checks 포함
uv run bird-xai-ws-smoke --with-negative --wish-interval 0
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--base-url` | `http://127.0.0.1:8080` | HTTP base (`/wish`, `/health`) |
| `--url` | (derived) | WS URL override |
| `--timeout` | `600` | step timeout (초) |
| `--frames` | `5` | wish 후 추가 frame 수 |
| `--with-wish` / `--no-with-wish` | on | override 시나리오 |
| `--with-negative` | off | 400/429 checks |
| `--flush-threshold` | `10` | expected `message_cnt` |
| `--wish-interval` | `0` | wish POST 간격 (CI: 0) |

## CI

GitHub Actions에서 smoke 실행 시 서버 env `BIRD_XAI_WISH_RATE_LIMIT_SEC=0` 권장.  
artifacts cache → [docs/deploy.md](../docs/deploy.md)

## 관련

- 서버 API: [ai/server/README.md](../ai/server/README.md)
- 배포: [docs/deploy.md](../docs/deploy.md)
