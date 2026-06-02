# ai/inference/

학습된 `BirdForecastLSTM`으로 **미래 12 step** 경로를 예측하고, step별 Captum IG XAI를 계산해 `FrameMessage`를 생성합니다.

## 주요 역할

- `BirdForecastLSTM` + scaler 로드 (`feat_scaler.pkl`, `delta_scaler.pkl`)
- 과거 24 step 입력 → 미래 12 point 절대 좌표 예측 (Δ 역변환 후 누적)
- 미래 horizon step별 IG attribution (0–1 정규화, 12개)
- IG 내부 특성: `ws_850`, `t_850`, `q_850`, `lapse_rate` → Unity 3키: `tailwind` / `headwind` / `weather_key` (`weather_key` ← `lapse_rate`)
- queue 기반 frame 생성 (관측 position + 12 future)
- `message_cnt` override → `ws_850` 입력 반영

## 내부 구성

- `pipeline.py` — `BirdPipeline` (예측 + XAI + queue + `build_frame_from_queue`)

## 파이프라인

```text
test_loader (batch=1 slice)
    ↓
BirdPipeline.build_queue(overrides?)
    ├── predict(X, last_obs)  → 12 future Point (절대 좌표)
    └── apply_xai(X)          → 12 XaiResult (미래 step별)
    ↓
queue: [(last_obs, xai_0), (f1, xai_0), (f2, xai_1), …, (f12, xai_11)]
    ↓
build_frame_from_queue() → FrameMessage
    (candidates=[], boids=null)
```

## 큐·refill

| 상수 / 동작 | 값·설명 |
|---|---|
| `QUEUE_REFILL_THRESHOLD` | `14` — 큐 길이가 이보다 작으면 백그라운드 refill (`service.py`가 스케줄) |
| override batch | `message_cnt` flush 시 `build_queue`로 **pending queue** 생성 → 다음 dequeue 시 swap |
| refill | 동일 `overrides`로 `build_queue` 후 기존 큐 **extend** |

세부 latency·운영 튜닝 → [docs/server-queue-latency.md](../../docs/server-queue-latency.md)

## Frame 출력

| 필드 | 값 |
|---|---|
| `position` | 관측 또는 예측 좌표 (절대 lat/lon/altitude_m) |
| `predicted_path` | 남은 미래 예측 point |
| `xai.attributions` | `tailwind`, `headwind`, `weather_key` |
| `candidates` | 항상 `[]` |
| `boids` | 항상 `null` (Unity 로컬 Boids) |
| `applied_overrides` | flush 시 `message_cnt` |

## 관련

- [ai/server/README.md](../server/README.md) — `iter_frames()`, wish flush
- [contracts/README.md](../../contracts/README.md) — Unity `WS /ws` JSON Schema
