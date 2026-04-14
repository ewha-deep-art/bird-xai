# data/

## 작업 범위

- `preprocessing/` — Kalman 필터(노이즈 제거), Cubic Spline 보간(균일 경로 생성), ERA5 기후 데이터 시간 매핑
- `raw/` — Movebank 원본 CSV 다운로드 및 보관
- `processed/` — 전처리 완료 데이터 저장

출력 포맷은 `contracts/data_schema.md` 참고.

## 주의

`raw/`, `processed/` 는 git에서 제외됩니다. 직접 다운로드하거나 팀 내 공유 경로를 이용하세요.
