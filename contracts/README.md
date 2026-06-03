# contracts/

Python 서버와 Unity 클라이언트가 공유하는 **WebSocket 계약** (`WS /ws`)입니다. 런타임 타입은 [`ai/common/models.py`](../ai/common/models.py), JSON Schema는 이 디렉토리의 `schemas/`·`examples/`가 기준입니다.

## 스키마

| 파일 | 설명 |
|---|---|
| `schemas/common.schema.json` | 공통 타입: `point`, `candidatePath`, `xaiResult`, `boidVelocity`, `boidAgent`, `attributionFeatureKey`, `overrideKey` |
| `schemas/frame.schema.json` | 서버 → Unity (`WS /ws`): 예측 윈도우 1 frame |
| `schemas/error.schema.json` | 서버 → Unity: 오류 메시지 |
| `schemas/server-message.schema.json` | WS 서버 메시지 union (`frame` + `error`) |
| `schemas/wish.schema.json` | HTTP `POST /wish` 요청·응답 (교차 검증용, 런타임은 `models.py`) |

## 예제 payload

| 파일 | 설명 |
|---|---|
| `examples/frame.sample.json` | `frame` 메시지 샘플 |

## 설계 원칙

- `position`, `predicted_path[]`는 모두 `point`
- `candidates`는 `[]`, `boids`는 `null`
- `xai.attributions`: `tailwind`, `headwind`, `weather_key` (IG 4특성 → `ws_850` 부호로 순풍·역풍, `weather_key` ← `lapse_rate`)
- `applied_overrides`: `message_cnt` (`OverrideKey`) — flush 시 frame에 포함, pipeline이 `ws_850` 입력에 반영
- XAI attribution: pipeline에서 0–1 정규화

## v1 고정값

| 항목 | 값 |
|---|---|
| schema version | `1.0.0` |
| window size | `24` steps (`ai/common/__init__.py`) |
| 학습 feature | `lat`, `lon`, `ground_speed`, `heading`, `ws_850`, `t_850`, `q_850`, `lapse_rate` |
| XAI feature key (Unity) | `tailwind`, `headwind`, `weather_key` |
| override key | `message_cnt` |
| frame `candidates` | `[]` |
| frame `boids` | `null` |

## `WS /ws`

연결 직후 `frame`을 `frame_interval` 초 간격으로 전송. 구조는 `schemas/frame.schema.json`.

## 검증

```bash
uv run bird-xai-server &
uv run bird-xai-ws-smoke
```
