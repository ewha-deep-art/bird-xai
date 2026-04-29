# contracts/

이 디렉토리는 Python 쪽 구현과 Unity 쪽 구현이 병렬적으로 진행될 수 있도록, 모듈 간 데이터 계약을 문서화하는 공간입니다. 특히 서버와 Unity 사이의 메시지 포맷, 추론 결과의 구조, 입력 이벤트 형식을 고정하는 데 사용합니다.

## 왜 필요한가

- Python과 Unity가 동시에 개발될 때 공통 기준점이 필요합니다.
- 서버 구현이 먼저 끝나지 않아도 Unity 쪽에서 mock 데이터를 만들 수 있어야 합니다.
- Unity 장면이 먼저 만들어져도 Python 쪽은 같은 스키마를 기준으로 payload를 맞출 수 있어야 합니다.

## 포함 문서 예시

- `frame-schema.json`
  - 프레임 단위 렌더링 데이터 구조
- `interaction-schema.json`
  - Unity에서 서버로 보내는 조작 입력 구조
- `xai-schema.json`
  - feature 기여도 표현 방식
- `README.md`
  - 각 스키마의 목적과 버전 규칙 설명

## 다루는 기술

- `JSON Schema`
  - 메시지 구조와 필수 필드 정의
- `Pydantic`
  - Python 쪽 검증 모델의 기준
- Unity `C# class` 또는 `struct`
  - 동일 계약을 Unity 쪽 타입으로 매핑

## 주요 계약 대상

- 전처리 산출물의 기본 필드 구조
- predictor 출력 형식
- XAI 결과 구조
- boids 시뮬레이션 결과 구조
- FastAPI WebSocket payload 형식
- Unity 인터랙션 입력 형식

프로젝트 전체 구조는 [README.md](../README.md), 서버 레이어는 [ai/server/README.md](../ai/server/README.md)를 참고하세요.
