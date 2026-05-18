# data/processed/
전처리 파이프라인의 산출물을 보관하는 디렉토리입니다.
LSTM 학습에 필요한 모든 파일이 이곳에 저장됩니다.

전처리 전체 코드: `BirdXAI_Preprocessing_LSTM.ipynb`

## 파일 목록

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `lstm_input_final.npz` | ~1MB | LSTM 학습용 최종 파일 (X, y) |
| `preprocessed_gps_era5.csv` | ~0.7MB | 전처리 완료 테이블 (검토·SHAP 분석용) |
| `feat_scaler.pkl` | - | feature MinMaxScaler (역변환용) |
| `tgt_scaler.pkl` | - | target MinMaxScaler (역변환용) |

---

## 대상 개체 (9마리, 1년차 어린 물수리)

| 이름 | 연도 | 이동 범위 | 지배 변수 | 전략 유형 |
|------|------|----------|----------|---------|
| Art | 2012 | 44°N → -6°N | 순풍지원 925hPa (r=0.541) | 순풍형 |
| Jill | 2012 | 44°N → 2°N | 순풍지원 925hPa (r=0.461) | 순풍형 |
| Bergen | 2013 | 39°N → -6°N | 광주기 (r=0.565) | 광주기형 |
| Whit | 2013 | 42°N → 18°N | 순풍지원 850hPa (r=0.612) | 순풍형 |
| Clyde | 2014 | 41°N → 11°N | 비습 850hPa (r=-0.422) | 날씨회피형 |
| Hudson | 2009 | 42°N → 6°N | 광주기 (r=0.431) | 광주기형 |
| Bea | 2009 | 41°N → 7°N | 광주기 (r=0.670) | 광주기형 |
| Caley | 2009 | 41°N → 5°N | 순풍지원 925hPa (r=0.366) | 순풍형 |
| Isabel | 2009 | 42°N → 8°N | 순풍지원 925hPa (r=0.465) | 순풍형 |

---

## 적용된 전처리

| 순서 | 항목 | 내용 |
|------|------|------|
| 1 | 활동 시간대 필터 | UTC 13~22시만 사용 (현지시간 9~18시, 물수리 이동 집중 시간대) |
| 2 | 이동일 필터 | displacement_km > 10km인 날만 사용 (이동 패턴 집중 학습) |
| 3 | log 변환 | `log(displacement_km + 1)` 적용 (이상치 1034km, 1039km 완화) |
| 4 | 광주기 정밀 계산 | `astral` 라이브러리로 위도+날짜 기반 일조시간 계산 |
| 5 | 개체 ID 원핫인코딩 | 9마리 개체 특성 차이를 모델에 반영 |
| 6 | 날짜 gap 처리 | 2일 초과 gap 구간의 윈도우 제외 |
| 7 | 개체별 슬라이딩 윈도우 | 개체 간 경계에서 윈도우 분리 후 병합 |
| 8 | 정규화 | MinMaxScaler — train 데이터로만 fit, 전체에 transform |

---

## lstm_input_final.npz 구조

### 로드 방법
```python
import numpy as np
data = np.load('lstm_input_final.npz')

X_train = data['X_train']  # (836, 7, 15)
y_train = data['y_train']  # (836,)
X_val   = data['X_val']    # (72,  7, 15)
y_val   = data['y_val']    # (72,)
X_test  = data['X_test']   # (152, 7, 15)
y_test  = data['y_test']   # (152,)
```

### Shape 설명

```
(샘플 수, 윈도우=7일, feature 수=15)
```

| 축 | 의미 |
|----|------|
| axis=0 | 샘플 수 |
| axis=1 | 연속 7일치 시퀀스 (윈도우 크기) |
| axis=2 | 15개 feature |

### Feature 순서 (axis=2)

| 인덱스 | 컬럼명 | 설명 | XAI 역할 |
|--------|--------|------|---------|
| 0 | `daylength_h` | 광주기 — 일조시간 (h) | **XAI 핵심** |
| 1 | `ws_925` | 순풍지원 925hPa (m/s) | **XAI 핵심** |
| 2 | `q_850` | 비습 850hPa (kg/kg) | **XAI 핵심** |
| 3 | `displacement_km` | 전날 이동거리 (km) | 관성 패턴 |
| 4 | `t_850` | 기온 850hPa (K) | 보조 |
| 5 | `lapse_rate` | 대기불안정도 t_925-t_700 (K) | 보조 |
| 6 | `bird_Art` | 개체 원핫인코딩 | 개체 특성 |
| 7 | `bird_Bea` | 개체 원핫인코딩 | 개체 특성 |
| 8 | `bird_Bergen` | 개체 원핫인코딩 | 개체 특성 |
| 9 | `bird_Caley` | 개체 원핫인코딩 | 개체 특성 |
| 10 | `bird_Clyde` | 개체 원핫인코딩 | 개체 특성 |
| 11 | `bird_Hudson` | 개체 원핫인코딩 | 개체 특성 |
| 12 | `bird_Isabel` | 개체 원핫인코딩 | 개체 특성 |
| 13 | `bird_Jill` | 개체 원핫인코딩 | 개체 특성 |
| 14 | `bird_Whit` | 개체 원핫인코딩 | 개체 특성 |

### 타깃 (y)
- 값: `log(displacement_km + 1)` 후 MinMaxScaler 정규화
- 범위: 0.0 ~ 1.0
- 의미: 다음 날 이동거리 예측

### 역변환 방법 (예측값 → km 단위 복원)
```python
import joblib, numpy as np

tgt_scaler = joblib.load('tgt_scaler.pkl')
y_pred_km = np.exp(tgt_scaler.inverse_transform(y_pred.reshape(-1, 1))) - 1
```

---

## Train / Val / Test 분할

| 구분 | 개체 | 샘플 수 | 비고 |
|------|------|--------|------|
| Train | Art, Jill, Hudson, Bea, Caley, Isabel | 836 | 순풍형·광주기형 혼합 |
| Val | Whit | 72 | 순풍형 검증 |
| Test | Bergen | 152 | 광주기형 — 일반화 성능 검증 |

> 시계열 특성상 랜덤 분할 금지. 개체 기준 분할 적용.  
> Bergen을 Test로 사용한 이유: Train과 가장 다른 전략(광주기형)을 가진 개체로,
> 모델이 순풍형 새들로 학습하고도 광주기형 새를 예측할 수 있는지 일반화 성능 검증 가능.

---

## XAI 활용 방향 (SHAP)

SHAP으로 개체별 feature 기여도를 계산하면 아래와 같은 개체별 차이가 예상됩니다.

| 개체 | 예상 높은 SHAP 변수 | 전략 |
|------|-------------------|------|
| Art, Jill, Whit, Caley, Isabel | `ws_925`, `ws_850` | 바람을 기다려 이동 |
| Bergen, Bea, Hudson | `daylength_h` | 낮 길이 변화에 반응 |
| Clyde | `q_850` | 악천후 회피 |

Unity 시각화 매핑:
- `daylength_h` SHAP → 경로 색상 (따뜻한/차가운 색조)
- `ws_925` SHAP → 경로 선 굵기·속도
- `q_850` SHAP → 경로 투명도·흐림 효과

---

## 관련 링크
- 전처리 코드: [ai/training/preprocessing/](../../ai/training/preprocessing/)
- 학습 phase: [ai/training/README.md](../../ai/training/README.md)
- 데이터 디렉토리: [data/README.md](../README.md)
