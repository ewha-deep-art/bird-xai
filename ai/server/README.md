# ai/server/

Bird XAI의 전달 레이어입니다. FastAPI + WebSocket 기반으로 Unity와 통신합니다.

## 엔드포인트

- `GET /health` — backend 상태 및 subject 확인
- `GET /viewer` — 브라우저용 시각화 페이지
- `WS /ws` — Unity와의 양방향 메시지 채널

## WebSocket 흐름

연결 직후 서버가 `frame`을 주기적으로 전송하기 시작합니다.

```
connect → server: frame, frame, frame, ...  (frame_interval 초 간격)
```

Unity에서 `controls.set`을 보내면 이후 frame부터 `applied_overrides`가 반영됩니다.

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
uv run bird-xai-server
```

환경 변수: `BIRD_XAI_HOST`, `BIRD_XAI_PORT`

## 파일 구조

| 파일 | 역할 |
|---|---|
| `app.py` | FastAPI 앱 생성, WebSocket 엔드포인트 |
| `service.py` | ML 파이프라인 연동 서비스 (real 모드) |
| `mock_service.py` | 정적 픽스처 반환 서비스 (mock 모드) |
| `session.py` | 세션 상태 (`subject_id`, `overrides`) |

## 관련 디렉토리
- 계약 및 스키마: [contracts/README.md](../../contracts/README.md)
- 추론 파이프라인: [ai/inference/README.md](../inference/README.md)