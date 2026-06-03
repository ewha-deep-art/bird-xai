# Bird XAI

![idea preview](docs/idea_v2.png)

철새 GPS·기후 데이터로 **다음 비행 경로**를 예측하고, Captum Integrated Gradients(IG)로 **순풍·역풍·날씨가 경로에 미친 영향**을 색으로 보여 주는 인터랙티브 미디어아트입니다.  
관람객은 스마트폰에서 「바람」을 보내고, 벽면 전시 화면에서 무리의 이동과 색 변화를 봅니다.

---

## 체험하기

전시장이 아니어도 **아래 두 링크**만 있으면 됩니다. 단계별 체크리스트·주의사항은 **[self_demo.md](self_demo.md)** 를 따르세요.

| 화면 | 링크 | 역할 |
|---|---|---|
| **벽면·큰 화면** (전시) | https://ewha-deep-art.github.io/bird-xai/ | 철새 무리·경로·XAI 색 — **보기만** |
| **스마트폰** (관람) | https://bird-xai-production.up.railway.app/wind-and-wish | 짧은 메시지 전송 — 전시장 QR과 동일 |

**3줄로 이해하기**

1. 큰 화면을 켜 두고 움직이는 무리를 본다.
2. 스마트폰에서 **바람 보내기**를 누른다. (글 내용은 AI가 읽지 않고, **보낸 횟수**만 셉니다.)
3. 바람이 **10번** 모이면(5초 간격 제한 있음) 예측 입력이 바뀌고, 큰 화면의 경로·색이 **몇 초~1분 안**에 바뀔 수 있습니다. 혼자 10번을 끝까지 보내 보세요.

백엔드 상태 확인: `GET https://bird-xai-production.up.railway.app/health`

---

## 작품 구성 (한눈에)

```text
[관람] QR → /wind-and-wish → POST /wish (메시지 횟수)
                    ↓
[AI] LSTM 예측 + Captum IG → frame 큐
                    ↓
[전시] WebGL(GitHub Pages) + Unity(native) ← wss `/ws` (경로 + XAI)
```

| 레이어 | 무엇을 하나 |
|---|---|
| 데이터 | Movebank GPS + ERA5 기후 → 학습·추론용 CSV·scaler |
| AI | 과거 24 step → 미래 12 step 경로 + step별 IG 설명 |
| 서버 | FastAPI — 관람 웹, `/wish`, `/ws` frame 스트림 |
| 렌더 | Unity [`render/`](render/) — `BirdDataManager`가 `/ws` 수신, Boids는 로컬 |

---

## 개발자·기여자

### 프로젝트 흐름

1. **학습** — `data/processed` + LSTM direct multi-step forecast  
2. **추론** — Captum IG + queue 기반 `FrameMessage`  
3. **전달** — FastAPI `/ws` → Unity, `/wish` → `ws_850` override  

```text
Movebank GPS + ERA5
  → data/processed
  → Training (BirdForecastLSTM)
  → Inference (BirdPipeline)
  → Server (/ws frame, /wish, /wind-and-wish)
  → Unity (render/)
```

### 스택

- 데이터: Movebank, ERA5  
- AI: Python (`PyTorch`, `captum`)  
- 서버: FastAPI, WebSocket  
- 렌더: Unity 2022 LTS (`render/`, VFX·Boids)  

### 디렉토리

| 경로 | 설명 |
|---|---|
| [self_demo.md](self_demo.md) | 방문자용 체험 가이드 |
| [docs/unimplemented.md](docs/unimplemented.md) | 미구현·보류 (A/B/C/D) |
| [docs/deploy.md](docs/deploy.md) | Railway·Docker·CI |
| [data/README.md](data/README.md) | loader, processed 산출물 |
| [ai/README.md](ai/README.md) | 학습·추론·서버 |
| [contracts/README.md](contracts/README.md) | Unity **WS** JSON Schema |
| [render/README.md](render/README.md) | Unity 클라이언트 |
| [tests/README.md](tests/README.md) | `bird-xai-ws-smoke` |
| [AGENTS.md](AGENTS.md) | 에이전트·구현 스냅샷 |

### 환경

```bash
uv sync --extra dev
```

| 명령 | 설명 |
|---|---|
| `bird-xai-train` | 학습 → `ai/training/model/weights/bird_best.pt` |
| `bird-xai-server` | API + 관람 웹 + `/ws` |
| `bird-xai-ws-smoke` | frame·wish smoke test |

`bird-xai-preprocess`는 entry만 등록됨 — `data/preprocess.py` 포팅은 [docs/unimplemented.md](docs/unimplemented.md) B항.

### 로컬 실행

런타임 4종(processed csv·pkl 2개, `bird_best.pt`)이 repo에 있으면 서버만으로 동작합니다. 없으면 [docs/deploy.md](docs/deploy.md) checklist 참고.

```bash
uv run bird-xai-train --epochs 1   # 가중치 재학습 시
uv run bird-xai-server
```

- 관람 웹: http://127.0.0.1:8080/wind-and-wish  
- 배포·artifact checklist: [docs/deploy.md](docs/deploy.md)  

### 빠른 검증

```bash
BIRD_XAI_WISH_RATE_LIMIT_SEC=0 uv run bird-xai-server &
uv run bird-xai-ws-smoke --wish-interval 0
```

Production smoke 예:

```bash
uv run bird-xai-ws-smoke \
  --base-url https://bird-xai-production.up.railway.app \
  --url wss://bird-xai-production.up.railway.app/ws
```

### 참고

- 구현 규칙·고정값(window 24, horizon 12 등): [AGENTS.md](AGENTS.md)  
- 일부 대용량 `data/raw/`는 Git 미포함  

### 라이선스 · 기여

| 문서 | 설명 |
|---|---|
| [LICENSE](LICENSE) | MIT License |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 브랜치·PR·smoke 검증·문서 갱신 |
| [docs/Team_Ground_Rule.md](docs/Team_Ground_Rule.md) | 팀 협업·코드 리뷰 규칙 |
