# ai/server/

이 디렉토리는 Python 기반 추론 결과를 Unity에 전달하는 서버 레이어입니다. 예측 경로, XAI 결과, 군집 상태를 하나의 메시지 흐름으로 묶어 실시간 렌더링에 적합한 형태로 전송합니다.

## 역할

- 추론 결과를 WebSocket 프레임 데이터로 변환
- Unity 클라이언트와 세션 연결 관리
- 관객 인터랙션 입력 수신
- 입력 변경 시 필요한 추론 파이프라인 재실행 연결

## 연결되는 입력

- `ai/inference/`의 경로 예측 결과
- `ai/inference/`의 feature 기여도
- `ai/inference/`의 군집 상태
- Unity 또는 외부 인터랙션 장치가 보낸 조작 값

## 사용할 기술

- `FastAPI`
- `Pydantic`
- Python `asyncio`
- `WebSocket`

## 출력

- WebSocket 기반 실시간 프레임 데이터
- 상태 갱신 이벤트
- 계약 문서에 맞춘 JSON payload

메시지 형식은 [contracts/README.md](../../contracts/README.md), 렌더링 소비자는 [render/README.md](../../render/README.md)를 참고하세요.
