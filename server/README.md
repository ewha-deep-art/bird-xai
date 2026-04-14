# server/ — FastAPI WebSocket 서버

## 역할

Python AI 파이프라인과 Unity 렌더링 사이의 통신을 담당합니다.
30fps로 프레임 데이터를 Unity에 스트리밍하고, Unity의 인터랙션 입력을 받아 모델에 전달합니다.