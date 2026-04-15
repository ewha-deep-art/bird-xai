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
### 1. 시공간 통합 매핑 (Spatio-temporal Matching)
철새의 GPS 위치와 기상 데이터는 서로 측정 시점과 위치가 일치하지 않습니다. 이를 하나로 합치는 과정이 필요합니다.
* **공간 매핑**: 기상 데이터는 일정 간격의 격자(Grid) 형태이므로, 격자와 격자 사이에 있는 철새의 정확한 위치에 맞춰 주변 기상 값을 추정(보간)하여 할당합니다.
* **시간 매핑**: 철새의 이동 시간과 기상 데이터의 기록 시간을 동기화하여, 해당 시점의 바람과 기류 정보를 정확히 매칭합니다.
* **기준**: 시차 오류를 방지하기 위해 모든 데이터는 국제 표준시(UTC)를 기준으로 통일합니다.

### 2. 데이터 정제 및 노이즈 제거 (Data Cleaning)
AI 모델이 잘못된 비행 패턴을 학습하지 않도록 비정상적인 수치를 걸러냅니다.
* **기기 오류 제거**: GPS 수신 불안정으로 인해 발생한 명백한 오류 데이터나 관리자가 수동으로 이상치라 표시한 기록을 제거합니다.
* **물리적 한계 검증**: 쇠재두루미의 비행 능력을 벗어나는 비정상적인 속도(예: 시속 200km 이상)가 감지될 경우, 이를 기기 오류로 간주하고 분석에서 제외합니다.
* 
---

## ⚠️ 주의사항
* **시간대 일치**: ERA5는 **UTC** 기준입니다. CSV의 현지 시간과 혼동하지 않도록 주의하십시오.
* **결측치 처리**: 기상 데이터 범위를 벗어난 좌표(영역 밖)는 매핑 시 `NaN`이 발생하므로 사전에 필터링해야 합니다.
