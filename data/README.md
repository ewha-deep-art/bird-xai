# data/

원본 데이터와 전처리 결과를 저장합니다. 학습·추론은 `data/loader.py`와 `data/processed/` 산출물을 사용합니다.

## 데이터 출처

### Movebank

- 철새 GPS 이동 경로 (개체 ID, 시각, 위·경도, 고도, 속도 등)

### ERA5

- Copernicus Climate Data Store 기후 변수 (풍속·풍향, 기온, 기압 레벨 등)
- GPS 시각·위치에 맞춘 feature 매핑

## 디렉토리 구성

### `raw/`
외부에서 수집한 원본 데이터를 보관합니다.
용량 문제로 Git에 포함하지 않으며, Google Drive에서 관리합니다.
파일 목록 및 접근 링크는 [raw/README.md](raw/README.md)를 참고하세요.
```
raw/
├── movebank/    ← GPS 원본 (Movebank)
│   └── .gitkeep
├── era5/        ← ERA5 기후 데이터 (CDS)
│   └── .gitkeep
└── README.md
```

### `processed/`

전처리 산출물. inference·training이 직접 참조.

**Git·Docker에 포함** (Railway 배포용): `preprocessed_geese_full.csv`, `feat_scaler.pkl`, `delta_scaler.pkl`. 샘플·컬럼 명세는 [processed/README.md](processed/README.md). checklist → [docs/deploy.md](../docs/deploy.md).

| 파일 | 용도 |
|---|---|
| `preprocessed_geese_full.csv` | 학습·추론 dataset (`ai/common` `DATASET_PATH`) |
| `feat_scaler.pkl` | 입력 feature `MinMaxScaler` (train fit) |
| `delta_scaler.pkl` | Δ target `sklearn.preprocessing.StandardScaler` (clip 후 train fit, 열별) |

### `notebooks/`
데이터 수집 및 전처리에 사용된 코랩 노트북을 보관합니다.
[notebooks/README.md](notebooks/README.md)를 참고하세요.

### `loader.py`

| 단계 | 설명 |
|---|---|
| 슬라이딩 윈도우 | 과거 `window_size`(24) → 미래 `forecast_horizon`(12) |
| target | 절대 `lat/lon/height_raw` → step별 Δ (`absolute_to_delta`) |
| Δ clip | `±[2°, 2°, 200m]` — outlier가 scaler를 지배하지 않도록 |
| scaler | feature MinMax, Δ StandardScaler; train split만 fit 후 pkl 저장 |
| 출력 | `(X, y_delta, last_obs)` 텐서 — `train_loader` / `val_loader` / `test_loader` |

## 데이터셋 현황

| 데이터셋 | 종 | 개체 수 | 기간 | 상태 |
|---------|-----|--------|------|------|
| North Sea White-fronted Geese | Anser albifrons | 65마리 | 2014~2017 9~11월 | 전처리 완료 |

## CLI

| 명령 | 상태 |
|---|---|
| (loader) | `import ai.common` 시 `get_data_loader` — 학습·추론 공용 |
| `bird-xai-preprocess` | entry만 등록 — `data/preprocess.py`는 [docs/unimplemented.md](../docs/unimplemented.md) B항 |

## 관련 디렉토리

- 학습 코드: [ai/training/README.md](../ai/training/README.md)
- 프로젝트 개요: [README.md](../README.md)
