# data/
이 디렉토리는 프로젝트의 원본 데이터와 전처리 결과 데이터를 저장하는 공간입니다.
학습과 추론의 기반이 되는 파일들이 이곳에 모이며, 외부 출처에서 받은 데이터와
내부 파이프라인이 생성한 산출물을 명확히 구분해 관리합니다.

## 데이터 출처

### Movebank
- 용도: 철새 GPS 이동 경로 데이터 수집
- 데이터셋: *Osprey Bierregaard North and South America*
- 종: Pandion haliaetus (물수리)
- 제공 정보: 개체 식별자, 시각 정보, 위도/경도, 고도, 속도 등
- 관측 간격: 1시간
- 분석 기간: 2009~2014년 9~11월 (가을 이동기)
- 프로젝트 활용 방식:
  - 1년차 어린 물수리 9마리 선별
  - 개체별 비행 경로 추출 및 이동 단위 분리
  - 모델 학습용 시계열 입력 생성

### ERA5
- 출처: Copernicus Climate Data Store (CDS)
- 용도: GPS 경로에 대응하는 기후 변수 제공
- 시간 해상도: 1시간 간격 (00:00~23:00 UTC)
- 공간 해상도: 0.25° × 0.25°
- 기압면: 700 / 850 / 925 hPa
- 제공 정보:
  - U/V wind (동서·남북 풍속)
  - Vertical velocity (수직 기류)
  - Temperature (기온)
  - Geopotential (지오퍼텐셜)
  - Specific humidity (비습)
- 프로젝트 활용 방식:
  - GPS 위치·시각에 맞춘 1:1 환경 변수 매핑
  - 모델 학습과 XAI 설명의 주요 feature 제공

## 디렉토리 구성

### `raw/`
외부에서 직접 수집하거나 다운로드한 원본 파일을 보관합니다.

```
raw/
├── movebank/
│   └── Osprey_Bierregaard_North_and_South_America.csv
└── era5/
    └── era5_hourly_YYYY_MM.nc  (Google Drive 관리 — Git 미포함)
```

- 사람이 직접 관리하는 입력 데이터
- 가공하지 않은 상태 유지
- ERA5 파일은 용량(GB급)으로 인해 Git에 포함하지 않음
  → Google Drive 경로: `MyDrive/BirdXAI/era5_hourly/`

### `processed/`
전처리 파이프라인이 생성한 학습 및 추론용 산출물을 보관합니다.

```
processed/
├── README.md                  ← 전처리 상세 및 파일 구조 설명
├── preprocessed_gps_era5.csv  ← 전처리 완료 테이블 (검토·SHAP 분석용)
├── lstm_input_final.npz       ← LSTM 학습용 최종 파일
├── feat_scaler.pkl            ← feature MinMaxScaler
└── tgt_scaler.pkl             ← target MinMaxScaler
```

- `ai/training/preprocessing/`의 결과 저장 위치
- 전처리 재현은 `BirdXAI_Preprocessing_LSTM.ipynb` 참고

## 이 폴더에서 다루는 기술

- 파일 포맷: `CSV`, `NetCDF`, `NPZ`, `PKL`
- 주요 라이브러리: `pandas`, `numpy`, `xarray`, `scikit-learn`, `astral`

## 관련 디렉토리
- 전처리 코드: [ai/training/preprocessing/](../ai/training/preprocessing/)
- 학습 phase: [ai/training/README.md](../ai/training/README.md)
- 프로젝트 개요: [README.md](../README.md)