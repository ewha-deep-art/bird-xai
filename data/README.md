# 📂 data/

이 디렉토리는 프로젝트에 사용되는 이동 경로 데이터와 기후 데이터를 관리합니다.

## 🛠 작업 범위

  * **preprocessing/**: Kalman 필터(노이즈 제거), Cubic Spline 보간(균일 경로 생성), ERA5 기후 데이터 시간 매핑
  * **raw/**: Movebank 원본 CSV 다운로드 및 보관 (**Git 제외**)
  * **processed/**: 전처리 완료 데이터 저장 (**Git 제외**)

> [\!CAUTION]
> `raw/` 및 `processed/` 폴더는 Git에 업로드되지 않습니다. 직접 다운로드하거나 팀 내 공유 경로를 이용하세요.
> 출력 포맷은 `contracts/data_schema.md` 참고.

-----

## 🛰 데이터 다운로드 가이드

### 1\. Movebank 데이터 (이동 경로)

  * **접속 경로**: https://www.movebank.org/
  * **사이트 사용 매뉴얼**: https://www.movebank.org/cms/movebank-content/manual
  * **방법**:
    1.  좌측 패널 검색창에서 **"1000 Cranes. Mongolia."** 검색
    2.  검색 결과에서 해당 연구 클릭.
    3.  상단 메뉴의 **[Download]** -\> **[Download Data]** 선택.
    4.  **CSV 포맷**으로 다운로드하여 `data/raw/` 폴더에 저장.

### 2\. ERA5 기후 데이터
> [\!CAUTION]
> 수정 필요

  * **접속 경로**: [ECMWF CDS](https://cds.climate.copernicus.eu/)
  * **매핑 규칙**: Movebank의 `timestamp`와 좌표를 기준으로 기후 데이터(기온, 강수량 등)를 시공간 매핑합니다.

-----

## 📊 데이터 스키마 (Data Schema)

### 1\. Raw Data (Movebank 원본 핵심 컬럼)

| 컬럼명 | 설명 | 비고 |
| :--- | :--- | :--- |
| `timestamp` | 위치 측정 시간 (UTC) | 정렬 및 보간의 기준 |
| `location-long` | 경도 (Longitude) | WGS84 좌표계, 지도상의 X축 좌표 |
| `location-lat` | 위도 (Latitude) | WGS84 좌표계, 지도상의 Y축 좌표 |
| `individual-local-identifier` | 개체 식별 번호 | 예: H17-6330 |
| `ground-speed` | 이동 속도 (m/s) | 비행 중인지, 쉬고 있는 중인지 판별 |
| `heading` | 진행 방향 | 새가 어느 방향(방위각)으로 향하고 있는지 확인 |
| `visible` | 데이터 유효성 여부 | false일 경우 필터링 권장 |

-----

## 💡 데이터 전처리 팁 (Team Guide)

새로운 데이터를 처리할 때 아래 가이드를 준수하여 데이터 품질을 유지해 주세요.

### 1\. 시간대(Timezone) 관리 (중요)

  * Movebank 데이터에는 `timestamp`(UTC)와 `study-local-timestamp`(현지 시간)가 공존합니다.
  * **모든 전처리와 ERA5 매핑은 `timestamp`(UTC)를 기준으로 수행**해야 시차 오류를 방지할 수 있습니다.

### 2\. 이상치 및 노이즈 제거

  * **Visible 필터링**: `visible` 컬럼이 `false`이거나 `manually-marked-outlier`가 `true`인 행은 GPS 오류이므로 사전에 제거합니다.
  * **속도 기반 필터링**: `ground-speed`가 생물학적으로 불가능한 속도(예: 200km/h 이상)로 찍힌 지점은 Kalman 필터 적용 전 검토가 필요합니다.

### 3\. 보간(Interpolation) 주의사항

  * **Cubic Spline 적용 시**: 데이터 간격이 너무 벌어진 구간(예: 배터리 문제로 며칠간 끊긴 경우)에 무리하게 보간을 적용하면 경로가 왜곡됩니다.
  * **최대 허용 간격**: 데이터 사이의 공백이 **24시간 이상**일 경우, 해당 구간은 보간하지 않고 경로를 분리하는 것을 권장합니다.

### 4\. 고도 데이터 처리

  * `height-above-ellipsoid`는 타원체고이므로, 실제 지면 높이(DEM)와 비교 시 보정이 필요할 수 있습니다.
