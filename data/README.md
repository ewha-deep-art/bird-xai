# data/
이 디렉토리는 프로젝트의 원본 데이터와 전처리 결과 데이터를 저장하는 공간입니다.
학습과 추론의 기반이 되는 파일들이 이곳에 모이며, 외부 출처에서 받은 데이터와
내부 파이프라인이 생성한 산출물을 명확히 구분해 관리합니다.

## 디렉토리 구성 및 가이드
### 📂 Data Directory Guide

본 디렉토리는 프로젝트에서 사용되는 데이터의 생명주기(원시 데이터 수집부터 전처리 완료까지)를 관리합니다. 데이터 보안 및 용량 제한으로 인해 실제 데이터셋은 Git 추적에서 제외될 수 있습니다.

data/
├── README.md          <- 현재 가이드 문서
├── raw/               <- 원본 데이터 (Raw Data) 보관 폴더
└── processed/         <- 전처리 완료된 데이터 (Processed Data) 보관 폴더

### ⚠️ 대용량 데이터 다운로드 안내
LSTM 모델의 인풋으로 사용되는 대용량 데이터셋은 깃허브 용량 한계로 인해 공용 Google Drive를 통해 제공됩니다. 아래 링크에서 다운로드 받아 본 디렉토리(`data/processed/`) 아래에 위치시켜 주세요.
- **다운로드 링크:** [공용 구글 드라이브 링크 입력]
- **파일명:** `preprocessed_9birds_full.csv`

#### 전처리한 데이터 컬럼 명세 (`preprocessed_9birds_full.csv`)

| 컬럼명 | 데이터 타입 | 설명 |
| :--- | :--- | :--- |
| **species** | String | 조류 종(Species) 정보 |
| **device** | Integer | 데이터 수집 장비 ID |
| **date_time** | DateTime | 데이터 기록 일시 (YYYY-MM-DD HH:MM:SS) |
| **latitude** | Float | 위도 (Latitude) |
| **longitude** | Float | 경도 (Longitude) |
| **altitude** | Float | 고도 (Altitude) |
| **geometry** | String | 공간 분석용 기하학적 포인트 데이터 (POINT) |
| **v_component_of_wind_10m** | Float | 10m 고도에서의 바람의 남북(V) 성분 (ERA5 데이터) |
| **u_component_of_wind_10m** | Float | 10m 고도에서의 바람의 동서(U) 성분 (ERA5 데이터) |
| **temperature_2m** | Float | 지상 2m 온도 (ERA5 데이터) |
| **total_precipitation** | Float | 총 강수량 (ERA5 데이터) |

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


## 관련 디렉토리
- 전처리 코드: [ai/training/preprocessing/](../ai/training/preprocessing/)
- 학습 phase: [ai/training/README.md](../ai/training/README.md)
- 프로젝트 개요: [README.md](../README.md)