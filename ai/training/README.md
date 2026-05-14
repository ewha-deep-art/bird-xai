# ai/training/

이 디렉토리는 Bird XAI v1의 전처리와 학습 경로를 담습니다. 현재 최소 기능 구현은 raw 데이터 1개체를 기준으로 **strict preprocessing -> dataset artifact 생성 -> LSTM 학습 -> checkpoint / weight 저장**까지 이어집니다.

## 현재 구조

- `preprocessing/movebank.py`
  - Movebank CSV를 canonical record로 변환
  - 중복 제거, 좌표 결측 제거, `gap > 6h` segment 분리
  - `ground_speed <= 1 m/s` 제거
  - 추정 속도 `> 40 m/s` outlier 제거
  - Kalman-like smoothing + 15분 간격 resampling
- `preprocessing/era5.py`
  - ERA5 nearest-neighbor 매핑
  - fallback 없이 실제 ERA5 dataset만 허용

## 기본 산출물

전처리 실행 시 아래 artifact가 생성됩니다.

- `data/processed/canonical/h17_6330.canonical.json`
- `data/processed/features/h17_6330.features.json`
- `data/processed/datasets/h17_6330.windows.json`

각 JSON artifact는 `metadata`를 포함합니다. 최소 포함 필드는 아래와 같습니다.

- `subject_id`
- `schema_version`
- `window_profile`
- `feature_order`
- `record_count`
- `segment_count`
- `window_count` (dataset artifact)
- `created_at`
- `strict_era5`

현재는 dependency-light 부트스트랩을 위해 JSON을 기본 저장 형식으로 사용합니다. Parquet/NPZ는 이후 단계에서 확장합니다.

## 실행

전처리:

```bash
bird-xai-preprocess
```

## 현재 제한 사항

- v1 대상은 `H17-6330` 1개체
- 최소 기능 단계의 window profile은 `48 observed -> 12 predicted`
- 학습용 dataset에는 미래 `12 step`이 실제로 존재하는 구간만 포함되며 zero-padding을 사용하지 않음
