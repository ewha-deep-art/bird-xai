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

아래 3파일(+ repo 루트의 `bird_best.pt`)은 **Railway Docker 빌드**를 위해 git에 포함됩니다. 그 외 `processed/` 파일(예: `preprocessed_9birds_full.csv`)은 gitignore.

| 파일 | 용도 |
|---|---|
| `preprocessed_geese_full.csv` | 학습·추론 dataset (`ai/common` `DATASET_PATH`) |
| `feat_scaler.pkl`, `delta_scaler.pkl` | MinMax scaler (loader·pipeline) |

## 코드

- `loader.py` — 슬라이딩 윈도우, scaler fit/transform, `train_loader` / `val_loader` / `test_loader`

## 전처리 CLI (예정)

`bird-xai-preprocess` entry point는 [pyproject.toml](../pyproject.toml)에 등록되어 있으나 `data/preprocess.py`는 아직 없음.  
팀 노트북 파이프라인을 Python으로 포팅 예정 ([docs/unimplemented.md](../docs/unimplemented.md) B항).

## 관련 디렉토리

- [ai/training/README.md](../ai/training/README.md)
- [README.md](../README.md)
