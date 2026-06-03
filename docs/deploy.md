# 배포 runbook

Bird XAI v1 — **Railway 1서비스**, Docker. 관람 QR은 `/wind-and-wish`. 전시 Unity WebGL은 Railway `/` mount 없음(native + GitHub Pages).

API 상세 → [ai/server/README.md](../ai/server/README.md). URL 맵·미구현 → [unimplemented.md](./unimplemented.md).

## Production

| 항목 | 값 |
|---|---|
| Base | https://bird-xai-production.up.railway.app |
| 관람 | `/wind-and-wish` |
| API | `/wish`, `/ws` (wss://), `/health` |

```bash
curl -s https://bird-xai-production.up.railway.app/health
uv run bird-xai-ws-smoke \
  --base-url https://bird-xai-production.up.railway.app \
  --url wss://bird-xai-production.up.railway.app/ws
```

## Runtime artifacts (git + Docker COPY)

| 파일 | 경로 |
|---|---|
| 가중치 | `ai/training/model/weights/bird_best.pt` |
| dataset | `data/processed/preprocessed_geese_full.csv` |
| feature scaler | `data/processed/feat_scaler.pkl` |
| delta scaler | `data/processed/delta_scaler.pkl` |

```bash
ls -lh ai/training/model/weights/bird_best.pt \
       data/processed/preprocessed_geese_full.csv \
       data/processed/feat_scaler.pkl \
       data/processed/delta_scaler.pkl
```

`.gitignore`는 위 4파일(+ `*_sample.csv`)만 tracked. 나머지 `data/processed/*`는 제외.

## Docker (로컬)

```bash
docker build -t bird-xai .
docker run --rm -p 8080:8080 bird-xai
```

검증:

```bash
curl http://127.0.0.1:8080/health
uv run bird-xai-ws-smoke --base-url http://127.0.0.1:8080
```

## Railway env

| 변수 | 설명 |
|---|---|
| `PORT` | Railway가 주입 — `BIRD_XAI_PORT`와 동일 alias |
| `BIRD_XAI_FRAME_INTERVAL` | WS frame 간격(초), 기본 `1.0` |
| `BIRD_XAI_WISH_RATE_LIMIT_SEC` | `/wish` IP 간격, 기본 `5` |
| `BIRD_XAI_WISH_FLUSH_THRESHOLD` | flush 누적 횟수, 기본 `10` |

## CI (GHA smoke)

[`.github/workflows/smoke.yml`](../.github/workflows/smoke.yml) — runtime 4파일이 없으면 **cache miss로 fail**.

1. self-hosted 등에서 4 artifact를 준비한 뒤 `workflow_dispatch` + `save_artifacts_cache: true`로 cache seed (C항 — [unimplemented.md](./unimplemented.md)).
2. PR/push job: cache restore → `bird-xai-server` (`BIRD_XAI_WISH_RATE_LIMIT_SEC=0`) → `bird-xai-ws-smoke`.
