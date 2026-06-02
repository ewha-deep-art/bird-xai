# ai/inference/

학습된 `BirdForecastLSTM`으로 **미래 12 step** 경로를 예측하고, step별 Captum IG XAI를 계산해 `FrameMessage`를 생성합니다.

## 주요 역할

- `BirdForecastLSTM` + scaler 로드
- 과거 24 step 입력 → 미래 12 point 절대 좌표 예측 (Δ 누적)
- 미래 horizon step별 IG attribution (0–1 정규화, 12개)
- IG 내부 특성: `ws_850`, `t_850`, `q_850`, `lapse_rate` → Unity 3키: `tailwind`/`headwind`/`weather_key` (현재 `weather_key` ← `lapse_rate`, 재정규화 없음)
- queue 기반 frame 생성 (관측 position + 12 future)
- `message_cnt` override → `ws_850` 입력 반영

## 내부 구성

- `pipeline.py` — `BirdPipeline` (예측 + XAI + queue + frame 빌드)

## 파이프라인

```text
test_loader (batch=1 slice)
    ↓
BirdPipeline._build_queue()
    ├── predict(X, last_obs)  → 12 future Point (절대 좌표)
    └── apply_xai(X)          → 12 XaiResult (미래 step별)
    ↓
queue: [(last_obs, xai_0), (f1, xai_0), (f2, xai_1), …, (f12, xai_11)]
    ↓
build_frame_from_queue() → FrameMessage
    (candidates=[], boids=null)
```

## Frame 출력

| 필드 | 값 |
|---|---|
| `position` | 관측 또는 예측 좌표 (절대 lat/lon/height) |
| `predicted_path` | 남은 미래 예측 point |
| `xai.attributions` | 해당 frame step의 `tailwind`, `headwind`, `weather_key` (Unity 3키, 4특성 IG 그대로·재정규화 없음) |
| `candidates` | 항상 `[]` |
| `boids` | 항상 `null` (Unity 로컬 Boids) |
| `applied_overrides` | `message_cnt` ( `/wish` 누적 시) |

## 관련 디렉토리

- [ai/server/README.md](../server/README.md)
- [contracts/README.md](../../contracts/README.md)
