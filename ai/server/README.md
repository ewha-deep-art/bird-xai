# ai/server/

Bird XAI의 전달 레이어입니다. FastAPI + WebSocket으로 Unity에 frame을 보내고, HTTP로 관람 입력을 받습니다.

## 엔드포인트

| 메서드 | 경로 | 용도 |
|---|---|---|
| `GET` | `/health` | 파이프라인 기동 여부·`subject_id` |
| `GET` | `/wind-and-wish` | QR 관람 웹 (정적 HTML) |
| `POST` | `/wish` | 관람 메시지 → `message_cnt` 누적 |
| `WS` | `/ws` | Unity로 `frame` 주기 전송 |

### `GET /health`

```json
{ "status": "ok", "subject_id": "White-fronted Goose" }
```

기동 실패 시: `{ "status": "error", "detail": "..." }`

### `POST /wish` (v1)

요청: `{ "message": "..." }` (1–50자, 본문 저장 없음)

응답: `{ "status": "ok", "subject_id": "White-fronted Goose" }`

타입: [`ai/common/models.py`](../common/models.py) — `WishRequest`, `WishResponse`

운영 규칙:

- 비어 있거나 50자 초과 → **400**
- 동일 IP 5초 내 재요청 → **429** (`BIRD_XAI_WISH_RATE_LIMIT_SEC`, `0`이면 비활성)
- 전역 `message_cnt` 누적, **≥ flush threshold**(기본 10) → `ws_850` override → counter reset
- flush된 override는 **다음 batch swap + 이후 refill까지 유지**
- `/ws` 연결 전 wish도 **첫 batch**에 반영 (`parse_overrides` 1회)

```bash
curl -X POST http://127.0.0.1:8080/wish \
  -H 'Content-Type: application/json' \
  -d '{"message":"바람"}'
```

### `WS /ws`

연결 직후 서버가 `frame`을 `frame_interval` 초 간격으로 전송합니다. 수신 JSON은 사용하지 않습니다 (연결 유지용).

```
connect → server: frame, frame, frame, ...
```

`frame` 구조: [contracts/schemas/frame.schema.json](../../contracts/schemas/frame.schema.json)

## 관람 웹 (`GET /wind-and-wish`)

QR → `/wind-and-wish` → `POST /wish`. 한국어 미니멀 UI.

```bash
open http://127.0.0.1:8080/wind-and-wish
```

정적 파일: [`static/participate.html`](static/participate.html)

## 서버 → Unity 메시지

| 타입 | 설명 |
|---|---|
| `frame` | 예측 경로 + XAI (`candidates=[]`, `boids=null`) |
| `error` | `startup_error`, `bad_request`, `pipeline_error` |

## 실행

```bash
uv run bird-xai-server
```

| 변수 | 기본값 | 설명 |
|---|---|---|
| `BIRD_XAI_HOST` | `0.0.0.0` | bind host |
| `BIRD_XAI_PORT` / `PORT` | `8080` | bind port (Railway는 `PORT`) |
| `BIRD_XAI_FRAME_INTERVAL` | `1.0` | WS frame 간격(초) |
| `BIRD_XAI_WISH_RATE_LIMIT_SEC` | `5.0` | `/wish` IP당 최소 간격(초). `0`이면 비활성 |
| `BIRD_XAI_WISH_FLUSH_THRESHOLD` | `10` | flush 전 누적 wish 건수 |

## 파일 구조

| 파일 | 역할 |
|---|---|
| `app.py` | FastAPI, 라우트, WebSocket send loop |
| `service.py` | `BirdPipeline` 연동, `iter_frames()`, wish flush·refill 스케줄 |
| `rate_limit.py` | IP rate limit (`limits`) |
| `static/participate.html` | QR 관람 웹 |

## 배포·검증

- Runbook: [docs/deploy.md](../../docs/deploy.md)
- Smoke: [tests/README.md](../../tests/README.md) — CI에서는 `BIRD_XAI_WISH_RATE_LIMIT_SEC=0` 권장

## 관련

- [ai/inference/README.md](../inference/README.md)
- [contracts/README.md](../../contracts/README.md)
