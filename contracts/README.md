# contracts/

Python 서버와 Unity 클라이언트가 공유하는 **단일 계약 원본**입니다. 런타임 모델은 `ai/contracts/models.py`의 Pydantic 모델을 쓰며, 구조의 기준은 이 디렉토리의 JSON Schema입니다.

## 스키마

| 파일 | 설명 |
|---|---|
| `schemas/common.schema.json` | 공통 타입: `point`, `candidatePath`, `xaiResult`, `boidVelocity`, `boidAgent`, `attributionFeatureKey` |
| `schemas/frame.schema.json` | 서버 → Unity: 예측 윈도우에서 잘린 1 프레임 |
| `schemas/error.schema.json` | 서버 → Unity: 오류 메시지 |
| `schemas/server-message.schema.json` | 서버 메시지 전체 union (`frame` + `error`) |
| `schemas/interaction-event.schema.json` | Unity → 서버: `controls.set` (유일한 인터랙션) |
| `schemas/model-output.schema.json` | 내부 계약: predictor → server (Unity 미전달) |

## 예제 payload

| 파일 | 설명 |
|---|---|
| `examples/frame.sample.json` | `frame` 메시지 샘플 (mock 서버가 이 파일을 로드) |
| `examples/controls-set.sample.json` | `controls.set` 이벤트 샘플 |

## 설계 원칙

- `position`, `predicted_path[]`, `candidates[].points[]`는 모두 동일한 `point` 타입
- `xai.attributions` 키 = `controls.set.overrides` 키 = `applied_overrides` 키 (같은 feature 네임스페이스)
- `attributionFeatureKey` enum: `["wind_speed", "wind_direction"]` — 추후 확장 가능
- XAI attribution은 서버에서 0-1 정규화 후 전달

## v1 고정값

| 항목 | 값 |
|---|---|
| schema version | `1.0.0` |
| 단일 개체 | `H17-6330` |
| timestep | `15분 (900초)` |
| observed window | `48 steps` |
| predicted window | `12 steps` |
| candidate paths | `3개` |
| 조작 가능 feature | `wind_speed`, `wind_direction` |

## Mock 서버

ML 파이프라인 없이 Unity 연동 테스트 가능:

```bash
# 환경변수
BIRD_XAI_MOCK=true uv run bird-xai-server

# 플래그
uv run bird-xai-server --mock

# smoke test
uv run bird-xai-ws-smoke
```
