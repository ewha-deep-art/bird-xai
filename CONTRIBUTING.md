# 기여 가이드

Bird XAI에 기여해 주셔서 감사합니다. 이 문서는 **외부 기여자·팀원**이 같은 절차로 작업할 때의 최소 규칙입니다. 팀 내부 협업 상세는 [docs/Team_Ground_Rule.md](docs/Team_Ground_Rule.md)를 따릅니다.

## 시작하기

1. [README.md](README.md) — 체험 URL·로컬 실행
2. [AGENTS.md](AGENTS.md) — 구현 스냅샷·고정값
3. [docs/unimplemented.md](docs/unimplemented.md) — 미구현(B/C/D) 확인 후 작업 (완료된 것처럼 문서만 바꾸지 않기)
4. 변경 범위 README — 해당 폴더 `README.md` 갱신

```bash
uv sync --extra dev
```

## 브랜치·PR

- `main`에 직접 push하지 않고 **feature 브랜치 → Pull Request**를 권장합니다.
- PR 전에 로컬에서 서버·smoke를 돌려 주세요 (아래 검증).
- frame·API 계약 변경 시 `ai/common/models.py`, `contracts/schemas/`, `examples/`를 함께 맞춥니다.
- 비밀값(`.env`, API 키, Railway token)은 커밋하지 않습니다.

## 검증

런타임 artifact 4종이 있을 때:

```bash
BIRD_XAI_WISH_RATE_LIMIT_SEC=0 uv run bird-xai-server &
uv run bird-xai-ws-smoke --wish-interval 0
```

- 서버·API: [ai/server/README.md](ai/server/README.md)
- smoke 옵션: [tests/README.md](tests/README.md)
- CI·Docker·artifact: [docs/deploy.md](docs/deploy.md)

## 문서·미구현

| 변경 | 할 일 |
|---|---|
| 코드·CLI·env | 해당 폴더 README + 필요 시 루트 README |
| 팀 결정·범위 | [docs/unimplemented.md](docs/unimplemented.md) B/C/D 이동 |
| 배포·Railway | [docs/deploy.md](docs/deploy.md) |

## 데이터·라이선스

- **소스 코드**: [LICENSE](LICENSE) (MIT)
- **Movebank·ERA5 등 외부 데이터**는 각 제공처 이용 약관을 따릅니다. `data/raw/` 대용량 원본은 repo에 없을 수 있습니다.
- 학습용 `data/processed/`·`bird_best.pt`는 용량이 큽니다. 불필요한 바이너리·개인 메시지 샘플을 PR에 넣지 마세요.

## 문의

- 이슈·PR: [GitHub ewha-deep-art/bird-xai](https://github.com/ewha-deep-art/bird-xai)
- 팀 규칙·회의: [docs/Team_Ground_Rule.md](docs/Team_Ground_Rule.md)
