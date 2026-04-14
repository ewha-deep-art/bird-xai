# Bird Migration XAI

> *자연은 고정되지 않는다. AI의 판단도 마찬가지다.*

![](idea/idea.png)

철새의 이동을 AI가 계산한다. 그 경로는 하나처럼 보이지만, 실제로는 수십 개의 가능성이 겹쳐있다.  
이 작업은 결과만이 아니라 — 풍향이 경로를 어떻게 틀었는지, 기온이 고도를 어떻게 낮췄는지 —  
AI의 판단 과정 자체를 시각적 레이어로 드러낸다.

---

## What it does

실제 GPS 데이터를 기반으로 철새 군집의 이동을 시뮬레이션하고,  
XAI(Explainable AI)를 통해 각 환경 변수가 경로 결정에 미친 영향을 실시간으로 시각화한다.  
관객은 풍향과 기온을 직접 조작하며 AI가 경로를 어떻게 재계산하는지 목격한다.

---

## Stack

`Python` `PyTorch` `SHAP` `Boids` `Unity VFX Graph`  
Data — [Movebank](https://www.movebank.org)

---

## Team

| 영역 | 담당 |
|---|---|
| `ml/` 전체 | 컴공과 |
| `unity/` 렌더링·비주얼 디자인 | 영상예술학과 |
| `unity/` WebSocket 수신·파티클 파라미터 연결 | 컴공과 |
| 인터랙션 기획 | 공동 |

---

*컴퓨터공학 × 영상예술학 협업*
