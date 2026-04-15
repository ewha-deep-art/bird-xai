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
