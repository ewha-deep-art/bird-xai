# render/

Unity 기반 렌더링 레이어입니다. Python 서버 `/ws`에서 받은 frame을 시각화하고, Boids 군집은 Unity에서 로컬 계산합니다.

## 역할 (계획)

- `predicted_path`·`position` 경로 시각화
- XAI attribution을 색·밝기 등으로 표현
- Unity VFX Graph 파티클 / Boids (서버 `boids` 필드 미사용)
- QR 관람 웹 → `POST /wish` (Unity WS 인터랙션 아님)

## 현재 상태

- `render/` monorepo 유지 ([docs/unimplemented.md](../docs/unimplemented.md) B항)
- Unity C# 프로젝트·씬 **미구현**

## 서버에서 수신하는 frame 필드

- `position`, `predicted_path[]`
- `xai.attributions` (`tailwind`, `headwind`, `weather_key`)
- `applied_overrides.message_cnt` (선택)
- `candidates`: 항상 `[]`
- `boids`: 항상 `null`

## 사용 기술 (예정)

- Unity 2022 LTS, VFX Graph, C#
- WebSocket client → `WS /ws`

## 관련 문서

- [ai/server/README.md](../ai/server/README.md)
- [contracts/README.md](../contracts/README.md)
