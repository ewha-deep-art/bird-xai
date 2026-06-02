# Bird XAI — 구현 가이드

프로젝트 개요와 디렉토리 설명은 [README.md](README.md) 및 각 폴더의 `README.md`를 참고합니다.

## 구현 원칙

- 구현은 `training -> inference -> server -> render` 흐름을 기준으로 진행합니다.
- 각 phase는 독립적으로 개발 가능해야 하지만, 최종적으로는 하나의 실시간 파이프라인으로 연결되어야 합니다.
- Python과 Unity의 병렬 작업은 `contracts/` 스키마와 `ai/common/models.py`를 기준으로 맞춥니다.

## 레이어

Karpathy [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) 패턴을 코드 프로젝트에 맞게 단순화한 구조입니다.

| 레이어 | 경로 | 역할 |
|---|---|---|
| **원본 (truth)** | 코드, `contracts/schemas/`, `data/processed/` | 에이전트가 읽고 따르는 사실. 문서만 바꿔서 구현된 것처럼 쓰지 않음 |
| **위키 (docs)** | 각 폴더 `README.md`, 루트 `README.md` | 사람·에이전트가 읽는 요약. 코드 변경 후 갱신 대상 |
| **스키마 (이 파일)** | `AGENTS.md` | 작업 규칙, 참조 순서, 현재 상태 스냅샷 |
| **미구현 레지스트리** | [docs/unimplemented.md](docs/unimplemented.md) | A/B/C/D 분류된 미구현·보류·미조사 항목 |

세션마다 코드를 처음부터 재발견하지 말고, **이 파일 → [미구현 레지스트리](docs/unimplemented.md) → 해당 README → 코드** 순으로 읽습니다.

## 현재 상태

파이프라인: `data/loader` → `ai/training` → `ai/inference/pipeline` → `ai/server` → (Unity `render/`)

미구현·보류·미조사 항목은 **[docs/unimplemented.md](docs/unimplemented.md)** 에 A/B/C/D로 정리되어 있습니다. README·contracts와 코드가 다를 때 그 문서의 분류가 우선합니다.

| 영역 | 구현됨 (요약) |
|---|---|
| 데이터 | `data/loader.py`, `data/processed/` (csv, scaler pkl) |
| 학습 | `BirdForecastLSTM`, `train.py`, `experiment.py`, `model/weights/` |
| 추론 | `BirdPipeline`: predict + Captum IG XAI, queue 기반 frame |
| 서버 | FastAPI `/health`, `/wind-and-wish` (QR HTML), `/wish` (JSON v1, rate limit), `/ws` frame 스트림, `ServerService.iter_frames()` |
| 계약 | `contracts/schemas/`, `ai/common/models.py` |
| 렌더 | `render/` (monorepo, Unity 코드 예정) |
| 검증 | `tests/smoke_client.py` (`bird-xai-ws-smoke`, wish→override 시나리오) |
| 배포 | Dockerfile artifacts COPY, [docs/deploy.md](docs/deploy.md), GHA [smoke.yml](.github/workflows/smoke.yml) |

## 작업 전 참조 (Query)

질문·구현 전에 아래만 열면 됩니다. 전 repo grep은 마지막 수단.

1. **미구현 여부** — [docs/unimplemented.md](docs/unimplemented.md) (A/B/C/D)
2. **범위 README** — [README.md](README.md) 디렉토리 표에서 해당 폴더 README
3. **공통 상수** — `ai/common/__init__.py` (feature, window, 경로)
4. **메시지 타입** — `ai/common/models.py` (런타임 truth). `contracts/schemas/`는 교차 검증용
5. **CLI** — `pyproject.toml` `[project.scripts]`

## 작업 후 정리

| 작업 | 할 일 |
|---|---|
| 코드·설정 변경 | 해당 폴더 README 갱신 (스킬: `update-readmes`) |
| frame payload 변경 | `ai/common/models.py` + `contracts/schemas/` + `examples/` 동시 수정 |
| 새 CLI·env var | `pyproject.toml` 또는 `ai/config.py` + 루트 README |
| 팀 결정 변경 | [docs/unimplemented.md](docs/unimplemented.md) A/B/C/D 이동 |

## Lint (건강 점검)

주기적으로 또는 contracts/서버 작업 전 확인:

- [ ] [docs/unimplemented.md](docs/unimplemented.md) B/C와 README·contracts·코드가 일치
- [ ] `AttributionFeatureKey` / `OverrideKey` — models.py ↔ schema ↔ `/wish`·pipeline 동작
- [ ] README의 파일명·CLI·경로가 repo에 실제 존재

## 코드 기준 고정값

`ai/common/__init__.py`와 models.py가 우선. README·schema와 다르면 코드를 따릅니다.

| 항목 | 값 |
|---|---|
| window (입력) | `24` steps |
| forecast (출력) | `12` steps |
| 학습 target | Δlat, Δlon, Δheight (Unity 전송 시 절대 좌표로 누적 변환) |
| 학습 feature | `lat`, `lon`, `ground_speed`, `heading`, `ws_850`, `t_850`, `q_850`, `lapse_rate` |
| XAI feature key | `tailwind`, `headwind`, `weather_key` (IG 내부 4특성 → Unity 3키, ws_850 부호로 순풍/역풍, 날씨는 `weather_key` 1개) |
| override (전시) | `/wish` JSON → `message_cnt` 누적 (≥10 flush) → `ws_850` |
| 배포 | **Railway 1서비스** — Dockerfile, weights+processed COPY, [docs/deploy.md](docs/deploy.md) |
| XAI | Captum IG (`n_steps=10`), queue prefetch ([docs/unimplemented.md](docs/unimplemented.md) B) |
| dataset | `data/processed/preprocessed_geese_full.csv` |
| weights | `ai/training/model/weights/bird_best.pt` |

## 통합 기준

- `training` 가중치·scaler는 `inference`에서 그대로 로드 (`load_model_with_state`, joblib scaler)
- `inference`는 `FrameMessage`를 반환; `server`는 JSON 직렬화만 담당
- Unity 연동 전 `uv run bird-xai-ws-smoke`로 `/ws` frame 스트림 검증
- 관람 입력은 `POST /wish` (`message_cnt` → `ws_850`)

## CLI

```bash
uv sync --extra dev
uv run bird-xai-train --epochs 1
uv run bird-xai-server          # real pipeline
uv run bird-xai-ws-smoke        # WebSocket smoke test
```

`bird-xai-preprocess`는 B항(노트북→Python 포팅 예정). 구현 전 README의 preprocess 선행 순서에 의존하지 말 것.
