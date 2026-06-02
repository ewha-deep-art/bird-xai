# ai/server/

Bird XAI의 전달 레이어입니다. FastAPI + WebSocket 기반으로 Unity와 통신합니다.

## 엔드포인트

- `GET /health` — backend 상태 및 subject 확인
- `GET /wind-and-wish` — QR 관람 웹 (정적 HTML)
- `POST /wish` — 관람객 메시지 수신 (JSON body). 누적 시 `message_cnt` → `ws_850` override
- `WS /ws` — Unity로 `frame` 주기 전송

## WebSocket 흐름

연결 직후 서버가 `frame`을 주기적으로 전송합니다.

```
connect → server: frame, frame, frame, ...  (frame_interval 초 간격)
```

`frame` 메시지 구조는 `contracts/schemas/frame.schema.json` 참조.

## 관람객 입력 (`POST /wish` v1)

JSON body `{ "message": "..." }` (1–50자, 본문 저장 없음).

- 비어 있거나 50자 초과 → **400**
- 동일 IP 5초 내 재요청 → **429**
- 전역 `message_cnt` 누적, **≥10건 flush** → `ws_850` override → counter reset

```bash
curl -X POST http://127.0.0.1:8080/wish \
  -H 'Content-Type: application/json' \
  -d '{"message":"바람"}'
```

계약: `contracts/schemas/wish.schema.json`, `contracts/examples/wish.sample.json`

## 관람객 웹 (`GET /wind-and-wish`)

QR 코드 → `/wind-and-wish` → `POST /wish`. 한국어 미니멀 UI.

```bash
# 브라우저
open http://127.0.0.1:8080/wind-and-wish
```

정적 파일: [`static/participate.html`](static/participate.html) (목업 → [`docs/visitor_web_demo.png`](../../docs/visitor_web_demo.png))

## 메시지 타입 (서버 → Unity)

| 타입 | 설명 |
|---|---|
| `frame` | 예측 경로 + XAI. `candidates=[]`, `boids=null` |
| `error` | 서버 오류 (`startup_error`, `bad_request`, `pipeline_error`) |

## 실행

```bash
uv run bird-xai-server
```

환경 변수:

| 변수 | 기본값 | 설명 |
|---|---|---|
| `BIRD_XAI_HOST` | `0.0.0.0` | bind host |
| `BIRD_XAI_PORT` / `PORT` | `8080` | bind port (Railway는 `PORT`) |
| `BIRD_XAI_FRAME_INTERVAL` | `1.0` | WS frame 간격(초) |
| `BIRD_XAI_WISH_RATE_LIMIT_SEC` | `5.0` | `/wish` IP당 최소 간격(초). `0`이면 비활성 |
| `BIRD_XAI_WISH_FLUSH_THRESHOLD` | `10` | flush 전 누적 wish 건수 |

## 배포

배포 runbook → [docs/deploy.md](../../docs/deploy.md)

## 파일 구조

| 파일 | 역할 |
|---|---|
| `app.py` | FastAPI 앱, `/health`, `/wind-and-wish`, `/wish`, WebSocket |
| `service.py` | ML 파이프라인 연동, `iter_frames()`, `subject_id` |
| `rate_limit.py` | IP rate limit (`limits`, FastAPI `Depends`) |
| `static/participate.html` | QR 관람 웹 (단일 HTML) |

## 관련 디렉토리

- 계약 및 스키마: [contracts/README.md](../../contracts/README.md)
- 추론 파이프라인: [ai/inference/README.md](../inference/README.md)
- 배포: [docs/deploy.md](../../docs/deploy.md)
