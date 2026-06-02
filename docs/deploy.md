# 배포 runbook

Bird XAI v1 배포 (Railway 1서비스, Docker). `/wind-and-wish` QR UI 포함. Unity WebGL `/` mount는 별도 phase.

API 엔드포인트 상세 → [ai/server/README.md](../ai/server/README.md)

## 사전 checklist — runtime artifacts

아래 4파일은 **git에 포함**됩니다 (Railway가 GitHub clone → `docker build` 시 Dockerfile COPY). 로컬에서 다른 processed/pt는 gitignore로 제외.

| 파일 | 경로 |
|---|---|
| 모델 가중치 | `ai/training/model/weights/bird_best.pt` (~5.4MB) |
| 학습 데이터 | `data/processed/preprocessed_geese_full.csv` (~74MB) |
| feature scaler | `data/processed/feat_scaler.pkl` |
| delta scaler | `data/processed/delta_scaler.pkl` |

확인:

```bash
ls -lh ai/training/model/weights/bird_best.pt \
       data/processed/preprocessed_geese_full.csv \
       data/processed/feat_scaler.pkl \
       data/processed/delta_scaler.pkl
```

## Docker 로컬 빌드·실행

```bash
docker build -t bird-xai .
docker run --rm -p 8080:8080 bird-xai
```

검증:

```bash
curl http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/wind-and-wish | head -5
uv run bird-xai-ws-smoke --base-url http://127.0.0.1:8080
```

`.dockerignore`는 `.venv`, `render/`, `data/raw/` 등을 제외합니다. artifacts 4파일은 Dockerfile에서 **명시적 COPY**합니다.

## Railway 배포

1. [Railway](https://railway.app)에서 새 프로젝트 생성 → GitHub repo 연결
2. **Build**: repo root [`Dockerfile`](../Dockerfile) (Railway가 자동 감지)
3. **Port**: Railway가 `$PORT` 주입 → [`ai/config.py`](../ai/config.py)가 `PORT` / `BIRD_XAI_PORT` 모두 수용
4. **Env** (선택):
   - `BIRD_XAI_FRAME_INTERVAL=1.0`
   - `BIRD_XAI_WISH_RATE_LIMIT_SEC=5.0`
   - `BIRD_XAI_WISH_FLUSH_THRESHOLD=10`
5. **Health check**: `GET /health` → `{ "status": "ok", ... }`
6. Public URL smoke:

```bash
uv run bird-xai-ws-smoke \
  --base-url https://<railway-domain> \
  --url wss://<railway-domain>/ws
```

**v1에서 하지 않음**: WebGL `/` StaticFiles mount — Unity WebGL 빌드 준비 후 추가.

`/wind-and-wish`는 [`ai/server/static/participate.html`](../ai/server/static/participate.html)을 `GET /wind-and-wish`로 serve (Docker `COPY ai/`에 포함).

### Railway 운영 메모 (C항)

- URL: `*.up.railway.app` default OK (custom domain은 유료 tier)
- cold start·IG queue refill lag: 배포 후 실측 ([unimplemented.md](unimplemented.md) C «전시 latency·운영»)

## CI artifacts (GitHub Actions)

weights·processed 4파일은 repo에 커밋되어 Railway 빌드에 사용됩니다. CI는 동일 경로를 restore하거나 cache seed(선택)로 보강합니다.

### Tarball 구조

```text
bird-xai-artifacts.tar.gz
├── ai/training/model/weights/bird_best.pt
└── data/processed/
    ├── preprocessed_geese_full.csv
    ├── feat_scaler.pkl
    └── delta_scaler.pkl
```

로컬에서 생성:

```bash
tar czf bird-xai-artifacts.tar.gz \
  ai/training/model/weights/bird_best.pt \
  data/processed/preprocessed_geese_full.csv \
  data/processed/feat_scaler.pkl \
  data/processed/delta_scaler.pkl
```

### Cache seed (1회)

self-hosted runner 또는 artifacts가 checkout에 있는 환경에서:

1. GitHub → **Actions** → **Smoke test** → **Run workflow**
2. `save_artifacts_cache=true` 선택 후 실행
3. `bird-xai-artifacts-v1` cache key로 4파일 저장

이후 push/PR마다 cache restore → smoke 실행.

Cache miss 시 workflow는 **fail** + `docs/deploy.md` 안내 메시지.

## 관련

- 서버 API: [ai/server/README.md](../ai/server/README.md)
- Smoke test: [tests/README.md](../tests/README.md)
- 미구현 레지스트리: [unimplemented.md](unimplemented.md)
