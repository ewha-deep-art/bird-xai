# XAI Demo

MNIST 기반 딥러닝 모델(CNN, GAN, Diffusion)을 학습하고, 다양한 XAI(Explainable AI) 기법으로 시각화하는 프로젝트입니다.

## 요구 사항

- Python 3.11 이상
- [uv](https://docs.astral.sh/uv/) 패키지 매니저

## 설치

```bash
uv sync
```

## 실행 방법

### 전체 파이프라인 한 번에 실행 (`run_all.py`)

3가지 모드를 지원합니다.

**1. 로컬 학습 + 시각화** (기본값, GPU 권장)

```bash
uv run python run_all.py
```

**2. HuggingFace 다운로드 + 시각화** (GPU 없이 빠르게 실행)

```bash
uv run python run_all.py --load
```

CNN/GAN은 CPU로 학습하고, Diffusion은 `1aurent/ddpm-mnist` 모델을 HuggingFace에서 다운로드합니다.

**3. 시각화만 실행** (모델이 이미 준비된 경우)

```bash
uv run python run_all.py --viz-only
```

`models/` 디렉토리에 아래 파일 중 하나 이상이 존재해야 합니다.

| 모델 | 필요 파일 |
|------|-----------|
| CNN | `models/mnist_cnn.keras` |
| GAN | `models/generator.weights.h5` |
| Diffusion | `models/unet.weights.h5` 또는 `models/ddpm-mnist/` |

---

### 개별 스크립트 실행

#### 모델 학습 (로컬)

```bash
uv run python train/train_cnn.py        # MNIST CNN
uv run python train/train_gan.py        # DCGAN
uv run python train/train_diffusion.py  # DDPM U-Net
```

#### 모델 로드 (HuggingFace / CPU 최적화)

```bash
uv run python loaders/load_cnn.py        # CNN (CPU, ~2분)
uv run python loaders/load_gan.py        # GAN (CPU 학습)
uv run python loaders/load_diffusion.py  # DDPM (HF: 1aurent/ddpm-mnist 다운로드)
```

#### 시각화

```bash
uv run python viz/01_cnn_filters.py           # CNN 필터 & 활성화 맵
uv run python viz/02_gradient_attribution.py  # Gradient Attribution
uv run python viz/03_perturbation.py          # Occlusion / LIME
uv run python viz/04_embedding.py             # t-SNE / UMAP 임베딩
uv run python viz/05_gan.py                   # GAN 생성 이미지
uv run python viz/06_diffusion.py             # Diffusion 노이즈 제거 과정
uv run python viz/07_shap.py                  # SHAP (DeepExplainer / GradientExplainer)
```

시각화 결과는 `outputs/` 디렉토리에 PNG 파일로 저장됩니다.

---

## 데이터셋 변경

`data.py` 상단의 `DATASET` 변수 한 줄만 변경하면 전체 파이프라인이 바뀝니다.

```python
# data.py
DATASET = 'mnist'         # 기본값: 28×28 흑백, 숫자 0-9
# DATASET = 'fashion_mnist' # 28×28 흑백, 의류 10종
# DATASET = 'cifar10'       # 32×32 컬러, 10종
```

## 프로젝트 구조

```
xai/
├── run_all.py          # 전체 파이프라인 실행 진입점
├── arch.py             # 모델 아키텍처 정의 (CNN, DCGAN, U-Net)
├── data.py             # 데이터셋 로딩 유틸리티
├── model_loader.py     # 모델 로딩 & 전처리 유틸리티
├── train/              # 로컬 학습 스크립트
├── loaders/            # HuggingFace / CPU 학습 스크립트
├── viz/                # XAI 시각화 스크립트 (01~06)
├── models/             # 학습된 모델 저장 위치 (자동 생성)
└── outputs/            # 시각화 결과 이미지 저장 위치 (자동 생성)
```
