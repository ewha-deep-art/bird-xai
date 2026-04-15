# render/ — Unity 렌더링 및 인터랙션

## 역할

서버로부터 WebSocket으로 수신한 데이터를 VFX Graph로 실시간 시각화하고,
관객의 인터랙션을 처리해 서버에 전달합니다.

```
render/
├── Assets/
│   ├── Scenes/             메인 전시 Scene (환경 및 조명 설정)
│   ├── VFX/                시각화 (VFX Graph 에셋)
│   ├── Scripts/            통신 및 데이터 바인딩
│   ├── Materials/          URP 기반 셰이더 및 재질
│   └── Prefabs/            기러기 모델 및 주요 오브젝트
├── PostProcess             영상 후처리
├── Packages/               사용된 패키지 의존성 (URP, VFX Graph 등)
└── README.md    
```

Assets/ — 에셋 폴더

사용할 에셋을 정리합니다.

PostProcess/ — 영상 후처리

실시간 영상 후처리 방법을 정리합니다.

Packages/ - 패키지 의존성

사용된 패키지 의존성을 관리합니다.


## 인풋

```selected_paths```:선두 기러기가 최종적으로 선택한 경로의 좌표값<br>
```predicted_paths```:후보 경로 n개의 좌표군(xai)<br>
```shap_values```: 각 환경 변수(풍향, 풍속, 기류)의 현재 경로 기여도 수치(xai)<br>
```boids_params```: 보이드 알고리즘으로 계산된 군집의 상태<br>
```user_input```: 풍속값을 변경하기 위해 유저가 인터랙션한 데이터<br><br>

## 출력

```wind_speed```: 유저에 의해 결정된 풍속값<br>
```timestamp```: 실시간 데이터 전송을 위한 timestamp<br>
전시환경에 따라 대형 프로젝션 맵핑을 위한 후처리를 한 후 프로젝터에 영상을 넘긴다<br><br>


# 기술 사양 및 파이프 라인

## 데이터 매핑 구조

수신된 비정형 데이터는 VFX Property Binder 를 통해 VFX Graph의 내부 변수와 1:1로 매핑



## 데이터 송수신

FAST API로 처리하되, ONNX 형식 모델로 내보낼 수 있다면 유니티 내부의 AI 기능 활용



## 후처리

Touch Designer 등을 사용할 수 있음


