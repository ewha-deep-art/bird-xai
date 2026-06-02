# data/
이 디렉토리는 프로젝트의 원본 데이터, 전처리 결과, 데이터 처리 코드를 저장합니다.

## 디렉토리 구성

### `raw/`
외부에서 수집한 원본 데이터를 보관합니다.
용량 문제로 Git에 포함하지 않으며, Google Drive에서 관리합니다.
파일 목록 및 접근 링크는 [raw/README.md](raw/README.md)를 참고하세요.
```
raw/
├── movebank/    ← GPS 원본 (Movebank)
│   └── .gitkeep
├── era5/        ← ERA5 기후 데이터 (CDS)
│   └── .gitkeep
└── README.md
```

### `processed/`
전처리 파이프라인이 생성한 학습용 산출물을 보관합니다.
전체 파일은 용량 문제로 Git에 포함하지 않으며, 샘플 파일과 컬럼 명세는
[processed/README.md](processed/README.md)를 참고하세요.

### `notebooks/`
데이터 수집 및 전처리에 사용된 코랩 노트북을 보관합니다.
[notebooks/README.md](notebooks/README.md)를 참고하세요.

## 데이터셋 현황

| 데이터셋 | 종 | 개체 수 | 기간 | 상태 |
|---------|-----|--------|------|------|
| North Sea White-fronted Geese | Anser albifrons | 65마리 | 2014~2017 9~11월 | 전처리 완료 |

## 관련 디렉토리
- 학습 코드: [ai/training/README.md](../ai/training/README.md)
- 프로젝트 개요: [README.md](../README.md)