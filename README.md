# Bird XAI
철새 군집 이동을 활용한 XAI 시각화 미디어아트

> *자연은 고정되지 않는다. AI의 판단도 마찬가지다.*

![idea.png](idea/idea.png)

철새의 이동을 AI가 계산한다. 그 경로는 하나처럼 보이지만, 실제로는 수십 개의 가능성이 겹쳐있다.  
이 작업은 결과만이 아니라 — 풍향이 경로를 어떻게 틀었는지, 기온이 고도를 어떻게 낮췄는지 —  
AI의 판단 과정 자체를 시각적 레이어로 드러낸다.

관객은 환경 변수를 직접 조작하며 경로가 재계산되는 과정을 목격한다.  
SHAP 기여도가 실시간으로 갱신되며, AI가 왜 그 경로를 선택했는지가 수치로 드러난다.

---

## Pipeline

```
Movebank GPS  ──┐
                ├──▶  Preprocessing  ──▶  Predictor  ──▶  XAI  ──▶  Boids  ──▶  Server  ──▶  Unity
ERA5 Climate  ──┘
```

| 단계 | 입력 | 출력 |
|------|------|------|
| **Preprocessing** | GPS raw CSV · ERA5 기후 데이터 | 균일 경로 · 시간 정렬된 환경 데이터 |
| **Predictor** | 전처리 경로 · 환경 데이터 | 후보 경로 50개 · 경로별 확률 |
| **XAI** | 예측 결과 · 입력 feature | 프레임별 feature 기여도 (SHAP) |
| **Boids** | 예측 경로 (리더) | 파티클 군집 위치 |
| **Server** | 파티클 위치 · SHAP · 후보 경로 | WebSocket 30fps 스트림 |
| **Unity** | WebSocket 수신 · 관객 조작 입력 | 렌더링 · 조작값 서버 전송 |

---

## Structure

```
bird-migration-xai/
├── contracts/              팀 간 인터페이스 계약
│   ├── data_schema.md      전처리 출력 포맷 · 예측 모델 입출력
│   └── websocket_schema.md Server ↔ Unity 메시지 포맷
│
├── data/                   데이터 수집 및 전처리
│   ├── preprocessing/      Kalman 필터 · Spline 보간 · ERA5 시간 매핑
│   ├── raw/                Movebank 원본 CSV (git 제외)
│   └── processed/          전처리 완료 데이터 (git 제외)
│
├── model/                  AI 모델
│   ├── predictor/          경로 예측 (LSTM + Monte Carlo Dropout)
│   ├── boids/              군집 시뮬레이션 (Boids)
│   ├── xai/                설명 가능한 AI (SHAP)
│   ├── weights/            모델 가중치 (git 제외)
│   └── checkpoints/        학습 체크포인트 (git 제외)
│
├── server/                 FastAPI WebSocket 서버
├── render/                 Unity VFX Graph 렌더링 · 인터랙션
└── idea/                   기획 문서
```

---

## Stack

**Data**  
Movebank (철새 GPS) · ERA5 / Copernicus Climate Data Store (풍향·풍속·기온)

**AI / ML**  
PyTorch · LSTM · Monte Carlo Dropout · SHAP · filterpy · scipy

**Backend**  
Python · FastAPI · WebSocket

**Rendering**  
Unity · VFX Graph

---

## Team
*컴퓨터공학 × 영상예술학*

| 역할 | 담당 |
|------|------|
| 데이터 전처리 · FastAPI WebSocket | . |
| 예측 모델 · Boids 시뮬레이션 · XAI | . |
| Unity VFX Graph · 인터랙션 | . |
| 비주얼 디자인 | 영상예술학 |

