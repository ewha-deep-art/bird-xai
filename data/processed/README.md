# data/processed/

전처리 파이프라인 산출물. 학습·추론·Docker COPY가 이 경로를 사용합니다.

## Git·런타임 (repo / Railway)

| 파일 | Git | 용도 |
|---|---|---|
| `preprocessed_geese_full.csv` | 포함 (~74MB) | 학습·추론 dataset |
| `feat_scaler.pkl` | 포함 | 입력 feature `MinMaxScaler` |
| `delta_scaler.pkl` | 포함 | Δ target `StandardScaler` |
| `preprocessed_geese_sample.csv` | 포함 | 상위 100행 샘플 (구조 확인) |

확인: [docs/deploy.md](../../docs/deploy.md). 재생성·원본은 Google Drive + 노트북 — 아래.

> Google Drive: `MyDrive/data/lstm_input_geese/`  
> 전처리 노트북: [notebooks/BirdXAI_Geese_Preprocessing.ipynb](../notebooks/BirdXAI_Geese_Preprocessing.ipynb)  
> Python CLI `bird-xai-preprocess`: [docs/unimplemented.md](../../docs/unimplemented.md) B항

## 데이터 개요

- **대상**: North Sea White-fronted Geese 65마리 (2014~2017)
- **분석 기간**: 9~11월 (가을 이동기)
- **시간 단위**: 1시간
- **총 행 수**: 약 수십만 행

## 적용된 전처리

| 순서 | 항목 | 내용 |
|------|------|------|
| 1 | 1시간 단위 집계 | GPS 원본(15분 간격)을 1시간 블록으로 평균 집계 |
| 2 | heading 처리 | 정지 중 0값 → NaN · circular mean 집계 · NaN은 bearing으로 대체 |
| 3 | 이상치 처리 | height_raw 음수값 → 0 클리핑 |
| 4 | 1시간 리샘플링 | 전체 기간을 1시간 간격으로 재생성 |
| 5 | 혼합 보간 | lat/lon/height → spline(3차) · ground_speed/heading → linear |
| 6 | gap 처리 | 24시간 초과 gap → 보간하지 않고 NaN 유지 |
| 7 | ERA5 매칭 | GPS 위경도·시각 기준 가장 가까운 격자점 추출 (1시간 이내) |
| 8 | ERA5 보간 | 매칭 실패 구간 선형 보간 |
| 9 | 파생 변수 계산 | ws, wspeed, wdir, lapse_rate, is_moving |
| 10 | 보간 여부 표시 | is_interpolated_gps, is_interpolated_era5 |

## 컬럼 명세 (44개)

### 메타 (4개)

| 컬럼명 | 설명 | 비고 |
|--------|------|------|
| `bird` | 개체 이름 | 예: GWFG_2015_408 |
| `timestamp` | 날짜+시각 (UTC) | 1시간 단위 · 예: 2014-09-01 13:00:00 |
| `is_interpolated_gps` | GPS 보간 여부 | 0=실측, 1=보간 |
| `is_interpolated_era5` | ERA5 보간 여부 | 0=직접매칭, 1=보간 |

### y 타깃 (3개)

| 컬럼명 | 설명 | 단위 |
|--------|------|------|
| `lat` | 위도 | °N |
| `lon` | 경도 | °E |
| `height_raw` | 고도 | m · 해수면 기준 |

### GPS 행동 변수 (3개)

| 컬럼명 | 설명 | 단위 / 비고 |
|--------|------|------------|
| `ground_speed` | 지상 이동속도 | m/s |
| `heading` | 이동 방위각 | 0~360° · 0=북 |
| `is_moving` | 이동 여부 | 0=정지, 1=이동 (ground_speed > 1) |

### ERA5 원본 (24개)

| 컬럼명 | 설명 | 단위 | 비고 |
|--------|------|------|------|
| `u_850` / `u_925` / `u_1000` | 동서 풍속 | m/s | 양수=서풍 |
| `v_850` / `v_925` / `v_1000` | 남북 풍속 | m/s | 양수=남풍 |
| `w_850` / `w_925` / `w_1000` | 수직기류 | Pa/s | 음수=상승 |
| `t_850` / `t_925` / `t_1000` | 기온 | K | K-273.15=℃ |
| `z_850` / `z_925` / `z_1000` | 지오퍼텐셜 | m²/s² | ÷9.81=고도(m) |
| `q_850` / `q_925` / `q_1000` | 비습 | kg/kg | 수증기 절대량 |
| `r_850` / `r_925` / `r_1000` | 상대습도 | % | 포화도 기준 |
| `cc_850` / `cc_925` / `cc_1000` | 구름량 | 0~1 | 0=맑음, 1=흐림 |

### 파생 변수 (10개)

| 컬럼명 | 설명 | 단위 | 계산식 |
|--------|------|------|--------|
| `ws_850` / `ws_925` / `ws_1000` | 순풍지원 | m/s | u·sin(heading)+v·cos(heading) · 양수=순풍 |
| `wspeed_850` / `wspeed_925` / `wspeed_1000` | 전체 풍속 | m/s | √(u²+v²) |
| `wdir_850` / `wdir_925` / `wdir_1000` | 바람 방향 | ° | atan2(u,v) · 0~360° |
| `lapse_rate` | 대기불안정도 | K | t_925-t_850 · 클수록 불안정 |