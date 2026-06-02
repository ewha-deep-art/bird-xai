# contracts/

Python 서버와 Unity 클라이언트가 공유하는 **단일 계약 원본**입니다. 런타임 모델은 [`ai/common/models.py`](../ai/common/models.py)의 Pydantic 모델을 쓰며, 구조의 기준은 이 디렉토리의 JSON Schema입니다.

## 스키마

| 파일 | 설명 |
|---|---|
| `schemas/common.schema.json` | 공통 타입: `point`, `candidatePath`, `xaiResult`, `boidVelocity`, `boidAgent`, `attributionFeatureKey`, `overrideKey` |
| `schemas/frame.schema.json` | 서버 → Unity: 예측 윈도우에서 잘린 1 프레임 |
| `schemas/error.schema.json` | 서버 → Unity: 오류 메시지 |
| `schemas/wish.schema.json` | 관람객 → 서버: `POST /wish` 요청·응답 |
| `schemas/server-message.schema.json` | 서버 메시지 전체 union (`frame` + `error`) |

## 예제 payload

| 파일 | 설명 |
|---|---|
| `examples/frame.sample.json` | `frame` 메시지 샘플 (현재 runtime 형태) |
| `examples/wish.sample.json` | 관람객 `POST /wish` 요청·응답 예시 |

## 설계 원칙

- `position`, `predicted_path[]`는 모두 동일한 `point` 타입
- `candidates`는 항상 빈 배열 (`[]`). `boids`는 서버에서 `null` (Unity에서 로컬 Boids)
- `xai.attributions` 키 = `AttributionFeatureKey` (`tailwind`, `headwind`, `weather_key`). IG는 내부 4특성으로 계산 후 `ws_850` 부호로 순풍·역풍 분리; 날씨 1개는 `weather_key`로 고정 키 전송 (현재 소스: `lapse_rate`)
- `applied_overrides` 키 = `OverrideKey` (`message_cnt`). `/wish` 누적 메시지 수가 `ws_850` 입력에 반영됨
- XAI attribution은 pipeline에서 0–1 정규화 후 전달

## v1 고정값

| 항목 | 값 |
|---|---|
| schema version | `1.0.0` |
| window size | `24` steps (`ai/common/__init__.py`) |
| 학습 feature | `lat`, `lon`, `ground_speed`, `heading`, `ws_850`, `t_850`, `q_850`, `lapse_rate` |
| XAI feature key (Unity) | `tailwind`, `headwind`, `weather_key` |
| override key | `message_cnt` |
| frame `candidates` | 항상 `[]` |
| frame `boids` | 항상 `null` (Unity 로컬) |

## 서버 인터페이스

### 서버 → Unity (`WS /ws`)

연결 직후 `frame`을 `frame_interval` 초 간격으로 전송합니다. 구조는 `schemas/frame.schema.json` 참조.

### 관람객 → 서버 (`POST /wish`)

QR 웹페이지 등에서 메시지를 보내면 `message_cnt`가 누적되고, 이후 frame의 `applied_overrides` 및 `ws_850` 입력에 반영됩니다.

```bash
curl -X POST http://127.0.0.1:8080/wish \
  -H 'Content-Type: application/json' \
  -d '{"message":"바람"}'
```

## 검증

```bash
uv run bird-xai-server &
uv run bird-xai-ws-smoke
```
