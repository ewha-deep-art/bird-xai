# 📂 data/

이 디렉토리는 프로젝트의 핵심인 **쇠재두루미(Demoiselle Crane)의 이동 경로 데이터**와 **ERA5 고해상도 기후 데이터**를 관리합니다. 

## 🛠 작업 범위
* **preprocessing/**: Kalman 필터(노이즈 제거), Cubic Spline 보간(균일 경로 생성), ERA5 기후 데이터 시공간 매핑(4D Interpolation).
* **raw/**: 원본 데이터 보관 (**Git 제외**).
* **processed/**: 기상 정보가 결합된 학습용 최종 데이터셋 (**Git 제외**).

---

## 🛰 데이터 수집 가이드

### 1. Movebank 데이터 (이동 경로)
* **Study명**: "1000 Cranes. Mongolia."
* **접속 경로**: [Movebank](https://www.movebank.org/)
* **수집 방법**: CSV 포맷 다운로드 후 `data/raw/` 저장.
* **특이사항**: 쇠재두루미(H17-6330 등)의 다년치 GPS 위치, 고도, 속도 정보 포함.

### 2. ERA5 기후 데이터 (환경 변수)
* **데이터셋**: `ERA5 hourly data on pressure levels from 1940 to present`
* **접속 경로**: [ECMWF CDS](https://cds.climate.copernicus.eu/)
* **다운로드 설정**:
    * **Product type**: `Reanalysis` (가장 정확한 역사적 기록 데이터)
    * **Temporal subset**: CSV 시계열 전체 범위인 **2018년 8월 ~ 2024년 12월** 선택 (용량 제한으로 인한 분할 다운로드 권장).
    * **Variables**: `U-component of wind' (동서 방향 풍속), 'V-component of wind' (남북 방향 풍속), `Vertical velocity` (수직 기류).
    * **Pressure Levels**: `700, 850, 925, 1000 hPa` (지표면~3,000m 상공 커버).

    * **Time Step**: 3시간 간격 (`00, 03, ..., 21`).
    * **Geographical Area**: `North: 50, West: 70, South: 20, East: 120` (몽골~인도 경로).
    * **Format**: `NetCDF4 (Experimental)`.

---

## 🔗 데이터 분석 및 전처리 링크
* **[Colab] ERA5 기상 데이터 추출 및 데이터 확인 (2018년 9-10월)**
    * [Google Colab 바로가기](https://colab.research.google.com/drive/1hdEx8ABRE8DJySuseejsWGdSO7E7dPLO?usp=sharing)
    * 내용: `.nc` 파일 구조 확인, 위경도/시간 범위 검증 및 Raw 데이터 샘플링.

---

## 📊 데이터 스키마 (Data Schema)

### 1. GPS Raw Data
| 컬럼명 | 설명 | 비고 |
| :--- | :--- | :--- |
| `timestamp` | 측정 시간 (UTC) | **매핑 기준 시간** |
| `location-long/lat` | 위경도 좌표 | WGS84 좌표계 |
| `ground-speed` | 지면 대비 속도 | 비행 여부 판별 (2m/s 이상 비행) |
| `height-above-ellipsoid` | 타원체 고도 (m) | 기압 레벨 매칭용 참조 데이터 |

### 2. Climate Data (NetCDF4 Experimental)
| 변수명 | 설명 | 단위 | 비고 |
| :--- | :--- | :--- | :--- |
| `valid_time` | 기상 유효 시간 | datetime64 | `time`에서 명칭 변경됨 |
| `pressure_level` | 기압 고도 | hPa | `level`에서 명칭 변경됨 |
| `u / v` | 수평 풍속 | m/s | 동서 / 남북 성분 |
| `w` | 수직 기류 속도 | **Pa/s** | 음수(-)가 강할수록 강한 상승기류 |

---

## 💡 데이터 전처리 및 매핑 가이드 (Technical Principles)

### 1. 시공간 매핑 (Spatio-temporal Matching)
철새 GPS 데이터는 측정 간격이 불규칙하며, ERA5 기상 격자와 일치하지 않는 좌표를 가집니다.
* **방법**: `xarray.Dataset.interp()` 함수를 사용하여 4차원(**valid_time, pressure_level, latitude, longitude**) 격자 데이터로부터 철새의 위치에 해당하는 기상 값을 **선형 보간(Linear Interpolation)**하여 추출합니다.
* **기준 시간**: 모든 매핑은 **UTC(`timestamp`)**를 기준으로 수행하여 시차 오류를 방지합니다.

### 2. 이상치 및 노이즈 제거 (Data Cleaning)
AI 모델의 학습 노이즈를 줄이기 위해 다음 데이터를 사전에 필터링합니다.
* **유효성 필터**: `visible: false` 또는 `manually-marked-outlier: true`인 행은 GPS 수신 오류이므로 제거합니다.
* **물리적 한계 필터**: `ground-speed`가 **200km/h(약 55.5m/s)**를 초과하는 데이터는 기기 오류로 간주하고 Kalman 필터 적용 전 제외합니다.

### 3. 고도(Pressure Level) 선정 근거
쇠재두루미의 독특한 비행 생태를 반영하여 4개의 기압 레벨을 사용합니다.
* **1000 ~ 925hPa**: 지표면 인근 기류 및 이착륙 환경 분석.
* **850hPa (약 1.5km)**: 철새의 주된 순항 고도이자 이동 경로 예측의 핵심 레이어.
* **700hPa (약 3.0km)**: **히말라야 산맥 횡단 대응.** 세계 최고 고도 비행 조류인 쇠재두루미가 험준한 산맥을 넘을 때의 고고도 기류 변화를 학습하기 위함입니다.

### 4. 보간(Interpolation) 및 경로 분리 규칙
데이터 연속성을 위해 Cubic Spline 보간을 적용할 때, 데이터의 왜곡을 막기 위한 **'데드라인'**을 설정합니다.
* **최대 허용 간격(24h)**: 데이터 사이의 공백이 **24시간 이상**일 경우, 무리하게 경로를 잇지 않고 경로 ID를 분리(Segmentation)합니다. 
* **이유**: 장시간 단절된 구간을 보간하면 실제 비행 패턴이 아닌 '가상의 직선 경로'를 학습하여 AI가 잘못된 비행 패턴을 학습하게 됩니다. 

---

## ⚠️ 주의사항
* **시간대 일치**: ERA5는 **UTC** 기준입니다. CSV의 현지 시간과 혼동하지 않도록 주의하십시오.
* **결측치 처리**: 기상 데이터 범위를 벗어난 좌표(영역 밖)는 매핑 시 `NaN`이 발생하므로 사전에 필터링해야 합니다.
