# render/

Unity 기반 렌더링 레이어입니다. Python 서버 `WS /ws`에서 frame을 받아 경로·XAI를 시각화하고, Boids 군집은 Unity에서 로컬 계산합니다.

## 역할

- `predicted_path`·`position` 경로 시각화 (VFX Graph 등)
- XAI attribution 색·밝기 (`tailwind`, `headwind`, `weather_key`)
- Boids 군집 — `frame.boids`는 항상 `null`, `BoidsController`가 로컬 시뮬레이션

## 구현 현황

| 구성 | 설명 |
|---|---|
| `Assets/Scripts/BirdDataManager.cs` | `ClientWebSocket` → `/ws` frame 수신·파싱·리더 이동 |
| `Assets/Scripts/BoidsController.cs` | 무리 Boids, `BirdDataManager`와 연동 |
| Production WS | `wss://bird-xai-production.up.railway.app/ws` (Inspector에서 host/port 설정) |

미구현(전시 UX): WS 끊김 시 **마지막 frame freeze**·**지수 backoff 재연결** — [docs/unimplemented.md](../docs/unimplemented.md) B항 「장애 UX」.

## WebGL 전시 (배포됨)

| 항목 | 설명 |
|---|---|
| 공개 URL | https://ewha-deep-art.github.io/bird-xai/ (GitHub Pages) |
| 백엔드 | `wss://bird-xai-production.up.railway.app/ws` — 빌드·`BirdDataManager`에서 production host 설정 |
| Railway `/` | v1에서 FastAPI StaticFiles WebGL mount **없음** — 전시는 Pages, API·관람은 Railway |

에디터·로컬 빌드는 `render/` Unity 프로젝트. 체험 순서 → [self_demo.md](../self_demo.md).

## 서버 frame 필드 (수신)

- `position`, `predicted_path[]`
- `xai.attributions` — `tailwind`, `headwind`, `weather_key`
- `applied_overrides.message_cnt` (flush 후)
- `candidates`: 항상 `[]`
- `boids`: 항상 `null`

JSON Schema: [contracts/schemas/frame.schema.json](../contracts/schemas/frame.schema.json)

## 사용 기술

- Unity 2022 LTS, VFX Graph, C#

## 관련

- [ai/server/README.md](../ai/server/README.md)
- [contracts/README.md](../contracts/README.md)
- [self_demo.md](../self_demo.md) — 전시·관람 URL
