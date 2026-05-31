# data/raw/
외부에서 수집한 원본 데이터를 보관하는 디렉토리입니다.
용량 문제로 Git에 포함하지 않으며, Google Drive에서 관리합니다.

## 파일 목록

| 파일명 | 설명 |
|--------|------|
| `North_Sea_population_tracks_of_greater_white-fronted_geese_2014-2017.csv` | GPS 원본 |
| `era5_geese/era5_geese_YYYY_MM.nc` | ERA5 기후 데이터 |


## 데이터 접근

### 팀원
팀 Google Drive에서 접근 가능합니다.

> 경로(철새): `내 드라이브 > data > North Sea population~`
> 경로(환경변수): `내 드라이브 > data > era5_geese`

### 외부 다운로드
원본 데이터는 아래 출처에서 직접 다운로드할 수 있습니다.
다운로드 설정은 아래 각 섹션을 참고하세요.

| 데이터 | 출처 | 링크 |
|--------|------|------|
| GPS (Movebank) | North Sea population tracks of greater white-fronted geese 2014-2017 | [Movebank](https://www.movebank.org) |
| ERA5 | ERA5 hourly data on pressure levels from 1940 to present
 | [CDS](https://cds.climate.copernicus.eu) |


## GPS 데이터

- **출처**: Movebank — *North Sea population tracks of greater white-fronted geese 2014-2017*
- **논문**: Kölzsch et al. 2019
- **종**: Anser albifrons (흰이마기러기)
- **개체 수**: 65마리 (2014: 13, 2015: 10, 2016: 20, 2017: 22)
- **전체 관측 기간**: 연중 (본 프로젝트에서는 9~11월 가을 이동기만 사용)
- **관측 간격**: 중앙값 0.3시간 (약 15분)
- **이동 경로**: 북극(76°N) → 유럽·중앙아시아 월동지

### 라이선스 및 인용

CC BY 또는 CC BY-NC 라이선스 데이터는 인용이 필수입니다. 
데이터 사용 시 아래 논문을 반드시 인용해주세요.
Kölzsch A, Müskens GJDM, Moonen S, Kraai A, Wikelski M, Nolet BA. 2019.
Goose migration along a leading line: wind, landform and snow cover rather
than true northward orientation drive spring migration across Siberia.
Royal Society Open Science. 6: 190208.
https://doi.org/10.1098/rsos.190208

데이터 소유자에게 프로젝트 사용 목적을 알리고 협업 가능 여부를 확인하는 것을 권장합니다. 

### Movebank 다운로드 설정
Study: North Sea population tracks of greater white-fronted geese 2014-2017
Format: CSV
Add UTM coordinates: 체크 안 함
Add study local time: 체크 안 함

## ERA5 기후 데이터

- **출처**: Copernicus Climate Data Store (CDS)
- **데이터셋**: ERA5 hourly data on pressure levels from 1940 to present
- **다운로드 코드**: [notebooks/ERA5_Download_NorthSea_Geese.ipynb](../notebooks/ERA5_Download_NorthSea_Geese.ipynb)

### 다운로드 설정

| 항목 | 값 |
|------|-----|
| Product type | Reanalysis |
| 변수 | U-component of wind, V-component of wind, Vertical velocity, Temperature, Geopotential, Specific humidity, Relative humidity, Fraction of cloud cover |
| 기압면 | 1000 / 925 / 850 hPa |
| 연도 | 2014, 2015, 2016, 2017 |
| 기간 | 9~11월 |
| 시간 | 00:00 ~ 23:00 (1시간 간격) |
| 영역 | N78 / W3 / S44 / E116 |
| 해상도 | 0.25° × 0.25° |
| 포맷 | NetCDF |

### 기압면 선택 근거
거위 비행 고도 분석 결과 (이동 중 기준):
- 중앙값: 55m / p95: 879m / p99: 1,584m
- 이동의 95%가 900m 이하 → 700hPa(3,000m) 제외
- 1000hPa(~100m) 추가로 실제 비행 고도 커버