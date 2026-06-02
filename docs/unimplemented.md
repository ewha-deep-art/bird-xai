# 미구현 레지스트리

팀이 확정한 **미구현·보류·미조사** 항목 목록입니다.  
코드·README·contracts와 충돌할 때 이 문서의 분류(B/C/D)가 우선합니다.

**분류**

| 태그 | 의미 |
|---|---|
| **A** | 구현하지 않음 (결정 완료) — repo에 흔적 없음 |
| **B** | 구현 예정 — 사양·방향 결정 완료 |
| **C** | 구현 예정 — 추가 결정 필요 |
| **D** | 미파악 — 범위·요구 조사 필요 |

**전시 제약 (2026-06-01 확정)**  
1달 이내 배포·운영. 과제 요건상 **단일 공개 URL** (WebGL 전시 + QR 관람 + API).

마지막 갱신: 2026-06-02 (B항 `/wind-and-wish` QR 웹 구현 완료)

---

## A. 구현하지 않음

현재 A항 없음. 폐기된 기능(mock, `controls.set`, 후보 경로 전송, Python Boids 등)은 코드·README·contracts에서 제거됨.

---

## B. 구현 예정 — 결정 완료

| 항목 | 확정 사양 | 현재 gap |
|---|---|---|
| **Railway 1서비스 배포** | Railway 앱 1개 + repo `Dockerfile`. FastAPI가 API + 정적 serve. URL: `/` WebGL, `/wind-and-wish`, `/wish`, `/ws`, `/health` | Railway 프로젝트 미생성. WebGL `/` mount 없음. API Docker·runbook → [deploy.md](deploy.md) |
| **Unity WebGL 전시** | `/` WebGL mount. 네이티브 Unity는 개발용 | WebGL 빌드·`/ws` 클라이언트 미구현 |
| **장애 UX (degraded mode)** | WS 끊김 → **마지막 frame 1개 freeze**. WS **자동 재연결**(1→2→4→…s, max 30s). `/health` 폴링 없음 | WebGL 클라이언트 미구현 |
| **`bird-xai-preprocess`** | 노트북 → Python 포팅, entry point 유지 | `data/preprocess.py` 없음 |
| **Unity `render/` (monorepo)** | Unity → WebGL 빌드 산출 | C#·씬 없음 |

### 1서비스 URL 맵 (확정)

```text
https://<railway-domain>/
  /              Unity WebGL (전시)
  /wind-and-wish   QR 관람 웹
  /wish          POST JSON
  /ws            WebSocket (wss://)
  /health        GET
```

---

## C. 구현 예정 — 추가 결정 필요

| 항목 | 확정된 것 | 열린 질문 |
|---|---|---|
| **Railway 배포 세부** | 1 Docker 서비스, free tier, `BIRD_XAI_*` env, Dockerfile·[deploy.md](deploy.md) runbook. **프로젝트 미생성**. v1 URL은 `*.up.railway.app` OK | custom domain(유료) 적용 시점, cold start·sleep, RAM/CPU 한도 실측 |
| **전시 latency·운영** | 로컬: `build_queue` ~331s → 768 frames, dequeue 즉시. IG 유지 | Railway free tier에서 refill lag·cold start 허용 여부, `n_steps`/배치 추가 튜닝 필요 시 일정 |
| **GHA artifacts cache** | [`.github/workflows/smoke.yml`](../.github/workflows/smoke.yml) + [deploy.md](deploy.md) CI 절 | self-hosted seed 1회 필요. cache miss 시 CI fail |

---

## D. 미파악 — 조사 필요

현재 D항 없음. latency는 C「전시 latency·운영」으로 이동(로컬 벤치 완료, Railway 실측 남음).

---

## 갱신 규칙

- 팀 결정이 바뀌면 **항목을 B↔C↔D로 이동**하고 한 줄 근거를 적는다. 폐기 기능은 repo에서 제거하고 A 표에 두지 않는다.
- B/C 항목이 **구현 완료**되면 해당 표에서 **제거**하고 README·contracts를 같이 고친다. 남은 미결정 사항만 C/D에 둔다.
- D에서 조사가 끝나면 C 또는 B로 승격한다.
- `docs/project.md` 등 기획 문서는 raw source — 구현 스냅샷은 이 파일 + AGENTS.md + README tree.
- 에이전트는 미구현 여부 판단 시 **이 파일을 AGENTS.md 다음에** 읽는다.

## C→B 조사 메모 (2026-06-01)

| C 항목 | 결정 | 비고 |
|---|---|---|
| `/wish` API | JSON body, 50자 | **구현 완료** (2026-06-02) |
| `/wish` 운영 | 5s/IP, flush≥10, 본문 미저장 | **구현 완료** |
| 데이터·가중치 | Docker COPY 번들 | **구현 완료** — [deploy.md](deploy.md) |
| degraded 구현 | last 1 frame + WS backoff | /health 폴링 생략 |
| UI/UX | 한국어·미니멀 | **구현 완료** — [`participate.html`](../ai/server/static/participate.html) |
| 윈도우 narrative | v1 생략 | LSTM step=24를 「15분」으로 설명하는 전시 카피 — v1 미사용 |
| XAI | IG 유지 | queue로 1s frame, refill ~5분 백그라운드 |
| smoke·CI | wish 시나리오 + GHA | **구현 완료** — cache seed는 C항 |
| 설문 | v1 제외 | — |
| Railway URL | default OK, custom은 유료 시 | C 잔류 |

**로컬 벤치** (CPU): `build_queue` 331s, queue 768 frames, `build_frame_from_queue` 즉시.
