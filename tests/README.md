# tests/

이 디렉토리는 Bird XAI 서버에 대한 간단한 smoke test 및 WebSocket 통신 검증을 담당합니다.

현재 구현은 터미널 기반 WebSocket 클라이언트를 통해 서버와 실제 연결을 수행하고, frame 스트리밍 및 control override 반영 여부를 검증하는 구조입니다.

---

## 주요 역할

- WebSocket 연결 검증
- bootstrap frame 수신 확인
- schema validation 검증
- controls.set roundtrip 테스트
- override 적용 여부 확인
- frame stream 정상 동작 확인

---

## 내부 구성

- `smoke_client.py`
  - 터미널 기반 WebSocket smoke test 클라이언트

---

### Smoke test

```bash
uv run bird-xai-ws-smoke
```

테스트 순서:
1. `WS /ws` 연결
2. 초기 `frame` 수신 확인
3. `controls.set` 전송 후 `applied_overrides`가 포함된 `frame` 수신 확인
4. 연속 frame stream 확인

옵션:

```bash
uv run bird-xai-ws-smoke --wind-speed 0.2 --wind-direction 0.1
uv run bird-xai-ws-smoke --url ws://127.0.0.1:8000/ws --timeout 10
```