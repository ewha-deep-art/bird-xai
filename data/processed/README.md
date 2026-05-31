# 📂 Data Processing Guide (`data/processed`)

본 디렉토리는 ERA5 기상 데이터와 조류 GPS 데이터를 시공간적으로 결합하고, LSTM 모델의 입력으로 사용하기 위해 전처리를 완료한 최종 데이터셋을 관리합니다.

---

## 📊 데이터셋 요약 (Dataset Summary)

* **최종 파일명:** `preprocessed_9birds_full.csv` (대용량 파일로 공용 Google Drive에서 다운로드 필요)
* **총 행 수 (Total Rows):** 15,693건 (9마리 × 9~11월 × 1시간 단위)
* **평균 보간 비율:** 약 48% (각 개체별 GPS 관측 밀도에 따라 상이)
* **결측치(NaN) 현황:**
* `latitude`, `longitude`: Art (28건), Clyde (244건), Hudson (59건) -> 24시간을 초과하는 공백 구간
* `ERA5 기상 변수`: 0건 (시공간 매칭 및 보간 완료)



---

## 🛠️ 전처리 파이프라인 (Preprocessing Pipeline)

### 1. GPS 이상치 처리 (Outlier Handling)

* **고도 음수값 처리:** 센서 오차로 발생한 `height_raw` 변수의 음수 값을 `0`으로 클리핑(Clipping) 조치했습니다. (예: Whit 개체 15건)
* **비정상 고도 처리:** 정지 중 센서 오류로 판단되는 비정상적인 고도 데이터 1건을 결측치(`NaN`) 처리했습니다. (예: Bergen 개체 2013-09-05 16:00의 7,100m 데이터)

### 2. 1시간 단위 리샘플링 (1-Hour Resampling)

* GPS 원본 관측 간격은 불규칙적입니다. (중앙값 1시간, 최대 95시간)
* 시계열 연속성을 위해 `pd.date_range(freq='1h')`를 사용하여 매 시간 정각 단위로 포인트를 생성하고, 실제 관측값이 없는 시간은 아래의 혼합 보간법을 적용했습니다.

### 3. 구간별 혼합 보간 (Mixed Interpolation)

이동 경로의 특성을 고려하여 변수별로 최적의 보간 알고리즘을 다르게 적용했습니다.

* **3차 스플라인 보간 (Cubic Spline):** 이동 경로의 자연스러운 곡선을 반영하기 위해 `latitude`, `longitude`, `height_raw`에 적용했습니다. *(※ 보간 과정에서 발생한 고도 음수값은 다시 0으로 재클리핑하여 보정)*
* **선형 보간 (Linear):** 속도와 방향의 변화를 반영하기 위해 `ground_speed`, `heading`에 적용했습니다.
* **임계값 제한 (Gap Limit):** 너무 긴 구간은 추정이 불가능하므로, **24시간을 초과하는 공백(Gap)** 구간은 보간하지 않고 `NaN` 상태를 유지했습니다. (Clyde 개체의 최대 95시간 공백 등)

### 4. ERA5 기상 데이터 1시간 단위 매칭

* 각 GPS 포인트의 위도, 경도, 시각에 가장 인접한 ERA5 격자점(Grid Point) 데이터를 추출했습니다.
* 1시간 이내의 매칭만 유효한 것으로 처리했으며, 매칭되지 않은 시간대는 선형 보간(`Linear Interpolation`)을 통해 결측치를 모두 채웠습니다.

### 5. 파생 변수 계산 (Feature Engineering)

모델의 학습 성능을 높이기 위해 도메인 지식을 기반으로 다음 4가지 파생 변수를 추가했습니다.

* **순풍지원 성분 ($ws_{925}$):** 조류 비행 방향에 작용하는 바람 성분

$$ws_{925} = u_{925} \times \sin(\text{heading}) + v_{925} \times \cos(\text{heading})$$


* **대기불안정도 ($\text{lapse\_rate}$):** 고도에 따른 기온 감률

$$\text{lapse\_rate} = t_{925} - t_{700}$$


* **이동 여부 ($\text{is\_moving}$):** 대지속도(`ground_speed`)가 $1\text{ m/s}$ 초과 시 True(1), 이하 시 False(0)
* **일조시간 ($\text{daylength\_h}$):** `astral` 라이브러리를 활용하여 위도와 날짜를 기반으로 계산한 정밀 일조시간

### 6. 데이터 신뢰도 표시 (Data Flag)

보간된 데이터의 왜곡 영향을 분석할 수 있도록 보간 여부 플래그를 제공합니다.

* `is_interpolated_gps`: GPS 위치 정보가 보간된 행인 경우 `1`
* `is_interpolated_era5`: ERA5 환경 변수가 보간된 행인 경우 `1`

---

## 📊 최종 데이터 컬럼 명세 (`preprocessed_9birds_full.csv`)

| 컬럼명 | 데이터 타입 | 설명 | 비고 |
| --- | --- | --- | --- |
| **species** | String | 조류 종 정보 | - |
| **device** | Integer | 데이터 수집 장비 ID | - |
| **date_time** | DateTime | 데이터 기록 일시 | YYYY-MM-DD HH:MM:SS |
| **latitude** | Float | 위도 (Latitude) | 3차 스플라인 보간 (24h 초과 제한) |
| **longitude** | Float | 경도 (Longitude) | 3차 스플라인 보간 (24h 초과 제한) |
| **height_raw** | Float | 고도 (Altitude) | 음수 0 클리핑, 스플라인 보간 |
| **ground_speed** | Float | 대지 속도 | 선형 보간 |
| **heading** | Float | 이동 방향 (방위각) | 선형 보간 |
| **u_component_of_wind_10m** | Float | 10m 고도 동서 바람 (U) | ERA5 기상 데이터 |
| **v_component_of_wind_10m** | Float | 10m 고도 남북 바람 (V) | ERA5 기상 데이터 |
| **temperature_2m** | Float | 지상 2m 기온 | ERA5 기상 데이터 |
| **total_precipitation** | Float | 총 강수량 | ERA5 기상 데이터 |
| **ws_925** | Float | 925hPa 고도 순풍지원 성분 | 파생 변수 (수식 참고) |
| **lapse_rate** | Float | 925hPa - 700hPa 기온 감률 | 파생 변수 (대기불안정도) |
| **is_moving** | Boolean | 이동 여부 플래그 | `ground_speed > 1 m/s` |
| **daylength_h** | Float | 정밀 계산된 일조시간 | `astral` 라이브러리 활용 |
| **is_interpolated_gps** | Binary | GPS 보간 여부 | 보간 시 1, 원본 데이터 시 0 |
| **is_interpolated_era5** | Binary | ERA5 보간 여부 | 보간 시 1, 원본 데이터 시 0 |

---

## 관련 링크
- 전처리 코드: [ai/training/preprocessing/](../../ai/training/preprocessing/)
- 학습 phase: [ai/training/README.md](../../ai/training/README.md)
- 데이터 디렉토리: [data/README.md](../README.md)
