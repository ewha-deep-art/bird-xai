# data/notebooks/
데이터 수집 및 전처리에 사용된 Google Colab 노트북을 보관합니다.

## 노트북 목록

| 파일명 | 설명 | 실행 순서 |
|--------|------|---------|
| `ERA5_Download_NorthSea_Geese.ipynb` | ERA5 기후 데이터 다운로드 | 1 |
| `BirdXAI_Geese_Preprocessing.ipynb` | GPS + ERA5 전처리 및 CSV 생성 | 2 |

## 실행 환경
- Google Colab
- Google Drive 연결 필요
- CDS API 키 필요 (ERA5 다운로드 시) → 발급: https://cds.climate.copernicus.eu/how-to-api

## 실행 순서

### 1. ERA5_Download_NorthSea_Geese.ipynb
ERA5 기후 데이터를 CDS에서 Drive로 다운로드합니다.

Drive 저장 경로: `MyDrive/data/era5_geese/`

파일 형식:
- 월 단위: `era5_geese_YYYY_MM.nc`
- 주 단위 (용량 초과 시 자동 분할): `era5_geese_YYYY_MM_wN.nc`

실행 방법:
- Cell 1~3: 최초 1회 실행 (설치, Drive 연결, API 키 입력)
- Cell 4~7: 연도별 순차 실행 (2014 → 2015 → 2016 → 2017)
- Cell 8: 전체 현황 확인

### 2. BirdXAI_Geese_Preprocessing.ipynb
GPS + ERA5 데이터를 매칭하고 전처리하여 최종 CSV를 생성합니다.

입력:
- `MyDrive/data/North_Sea_...csv` (GPS 원본)
- `MyDrive/data/era5_geese/` (ERA5 파일)

출력:
- `MyDrive/data/lstm_input_geese/preprocessed_geese_YYYY.csv` (연도별)
- `MyDrive/data/lstm_input_geese/preprocessed_geese_full.csv` (최종 합본)

실행 방법:
- Cell 1~3: 최초 1회 실행 (설치, Drive 연결, 함수 정의)
- Cell 4: YEAR 설정 (2014 → 2015 → 2016 → 2017)
- Cell 5~10: 연도별 반복 실행
- Cell 11: 모든 연도 완료 후 합치기

## 주의사항
- ERA5 다운로드는 CDS 서버 상태에 따라 수 시간 소요될 수 있음
- 세션 끊김 대비: 월별 중간 저장으로 이어받기 가능
- GPS 원본 및 ERA5 파일은 용량 문제로 Drive에서만 관리 (Git 미포함)
- 컬럼 명세 및 전처리 상세: [processed/README.md](../processed/README.md) 참고

## 관련 디렉토리
- 원본 데이터: [raw/README.md](../raw/README.md)
- 전처리 결과: [processed/README.md](../processed/README.md)