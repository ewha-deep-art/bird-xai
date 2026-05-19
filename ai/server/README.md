# ai/server/

Bird XAI의 전달 레이어입니다. FastAPI + WebSocket 기반으로 Unity와 통신합니다.

## 엔드포인트

- `GET /health` — backend 상태 및 subject 확인
- `GET /viewer` — 브라우저용 시각화 페이지
- `WS /ws` — Unity와의 양방향 메시지 채널

## WebSocket 흐름

연결 직후 서버가 `frame`을 주기적으로 전송하기 시작합니다.

```
connect → server: frame, frame, frame, ...  (BIRD_XAI_FRAME_INTERVAL 초 간격)
```

Unity에서 `controls.set`을 보내면 다음 frame부터 `applied_overrides`가 반영됩니다.

```
client: controls.set
                    → server: frame (applied_overrides 반영), frame, frame, ...
```

`frame` 메시지 구조는 `contracts/schemas/frame.schema.json` 참조.

## 메시지 타입

### 서버 → Unity

| 타입 | 설명 |
|---|---|
| `frame` | 예측 경로 + XAI + boids. 연결 직후 및 controls.set 응답 시 전송 |
| `error` | 서버 오류 (`startup_error`, `bad_request`, `pipeline_error`) |

### Unity → 서버

| 타입 | 설명 |
|---|---|
| `controls.set` | 풍속/풍향 override. `overrides: { wind_speed, wind_direction }` |

## 실행

```bash
# real 서버 (ML 파이프라인 필요)
uv run bird-xai-server

# mock 서버 (ML 파이프라인 불필요, Unity 연동 테스트용)
uv run bird-xai-server --mock
BIRD_XAI_MOCK=true uv run bird-xai-server
```

환경 변수: `BIRD_XAI_HOST`, `BIRD_XAI_PORT`, `BIRD_XAI_MOCK`, `BIRD_XAI_FRAME_INTERVAL` (기본 `1.0`초)

## Smoke test

```bash
uv run bird-xai-ws-smoke
```

테스트 순서:
1. `WS /ws` 연결
2. bootstrap `frame` 수신 확인
3. `controls.set` 전송 후 `applied_overrides`가 포함된 `frame` 수신 확인

옵션:

```bash
uv run bird-xai-ws-smoke --wind-speed 0.2 --wind-direction 0.1
uv run bird-xai-ws-smoke --url ws://127.0.0.1:8000/ws --timeout 10
```

## 파일 구조

| 파일 | 역할 |
|---|---|
| `app.py` | FastAPI 앱 생성, WebSocket 엔드포인트 |
| `service.py` | ML 파이프라인 연동 서비스 (real 모드) |
| `mock_service.py` | 정적 픽스처 반환 서비스 (mock 모드) |
| `session.py` | 세션 상태 (`subject_id`, `overrides`) |
| `smoke_client.py` | 터미널 WebSocket 테스트 클라이언트 |
