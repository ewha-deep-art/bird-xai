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

외부에서 받은 원본 파일 (Movebank CSV, ERA5 NetCDF). Git에는 최소화.

현재 repo에는 `.gitkeep`만 있을 수 있음. 실제 raw 파일은 로컬 또는 별도 저장소에서 관리.

### `processed/`

전처리 산출물. inference·training이 직접 참조.

| 파일 | 용도 |
|---|---|
| `preprocessed_geese_full.csv` | 학습·추론 dataset (`ai/common` `DATASET_PATH`) |
| `feat_scaler.pkl` | 입력 feature `MinMaxScaler` (train fit) |
| `delta_scaler.pkl` | Δ target `sklearn.preprocessing.StandardScaler` (clip 후 train fit, 열별) |

## 코드 (`loader.py`)

| 단계 | 설명 |
|---|---|
| 슬라이딩 윈도우 | 과거 `window_size`(24) → 미래 `forecast_horizon`(12) |
| target | 절대 `lat/lon/height_raw` → step별 Δ (`absolute_to_delta`) |
| Δ clip | `±[2°, 2°, 200m]` — outlier가 scaler를 지배하지 않도록 |
| scaler | feature MinMax, Δ StandardScaler; train split만 fit 후 pkl 저장 |
| 출력 | `(X, y_delta, last_obs)` 텐서 — `train_loader` / `val_loader` / `test_loader` |

`import ai.common` 시 `get_data_loader()`가 한 번 실행되며, 위 scaler pkl이 **train 데이터 기준으로 다시 fit·저장**됩니다. CSV만 바꿨을 때 pkl을 맞추려면 프로세스를 새로 띄우면 됩니다.

## 관련

- [ai/training/README.md](../ai/training/README.md)
- [README.md](../README.md)
