"""공유 모델 로딩 & 전처리 유틸리티 — 모든 viz 스크립트에서 import해 사용"""
import os
import numpy as np
import keras
import tensorflow as tf

ROOT   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, 'models')

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

# ─── 데이터 ───────────────────────────────────────────────────────────────────

def get_cifar10():
    """CIFAR-10 로드 → (x_train, y_train, x_test, y_test), x∈[0,1], y: 1D int"""
    (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
    x_tr = x_tr.astype('float32') / 255
    x_te = x_te.astype('float32') / 255
    return x_tr, y_tr.squeeze(), x_te, y_te.squeeze()

def get_mnist():
    """MNIST 로드 → (x_train, y_train, x_test, y_test), x∈[0,1]"""
    (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
    x_tr = np.expand_dims(x_tr.astype('float32') / 255, -1)
    x_te = np.expand_dims(x_te.astype('float32') / 255, -1)
    return x_tr, y_tr, x_te, y_te

# ─── ResNet-50 전처리 ──────────────────────────────────────────────────────────

def preprocess_resnet(imgs):
    """
    CIFAR-10 이미지 (H, W, 3) 또는 (N, H, W, 3), 값 [0, 1]
    → ResNet-50 입력 (N, 224, 224, 3), ImageNet 정규화
    """
    single = imgs.ndim == 3
    if single:
        imgs = imgs[np.newaxis]
    resized = tf.image.resize(imgs * 255, (224, 224)).numpy()
    processed = keras.applications.resnet.preprocess_input(resized)
    return processed[0] if single else processed

def deprocess_resnet(img):
    """
    ImageNet 정규화 역변환 → [0, 1] (표시용)
    ResNet preprocess: BGR 채널순, 각 채널에서 mean 뺌
    """
    img = img.copy().astype('float32')
    img[..., 0] += 103.939  # B
    img[..., 1] += 116.779  # G
    img[..., 2] += 123.68   # R
    img = img[..., ::-1]    # BGR → RGB
    return np.clip(img / 255, 0, 1)

# ─── 모델 로더 ─────────────────────────────────────────────────────────────────

def load_resnet50():
    """ResNet-50 (ImageNet pretrained) 로드"""
    path = os.path.join(MODELS, 'resnet50.keras')
    if os.path.exists(path):
        print(f'모델 로드: {path}')
        return keras.models.load_model(path)
    print('ResNet-50 다운로드 중 (ImageNet 가중치)...')
    return keras.applications.ResNet50(weights='imagenet', include_top=True)

def load_ddpm():
    """Google DDPM-CIFAR10-32 파이프라인 로드 (PyTorch diffusers)"""
    from diffusers import DDPMPipeline
    local = os.path.join(MODELS, 'ddpm-cifar10-32')
    if os.path.exists(local):
        print(f'DDPM 로드: {local}')
        return DDPMPipeline.from_pretrained(local)
    print('DDPM 다운로드 중 (google/ddpm-cifar10-32)...')
    return DDPMPipeline.from_pretrained('google/ddpm-cifar10-32')

def load_gan_generator():
    """학습된 DCGAN Generator 로드"""
    from arch import build_generator, LATENT_DIM
    gen = build_generator()
    gen(tf.zeros([1, LATENT_DIM]))   # build
    gen.load_weights(os.path.join(MODELS, 'generator.weights.h5'))
    return gen

# ─── 유틸 ─────────────────────────────────────────────────────────────────────

def class_samples(y, n_per_class=1, x=None):
    """클래스별 대표 인덱스 반환"""
    indices = []
    for c in range(10):
        idxs = np.where(y == c)[0][:n_per_class]
        indices.extend(idxs.tolist())
    return indices

def load_mnist_cnn():
    """MNIST CNN 로드 (로컬 우선, 없으면 자동 학습)"""
    path = os.path.join(MODELS, 'mnist_cnn.keras')
    if os.path.exists(path):
        print(f'모델 로드: {path}')
        return keras.models.load_model(path)
    print('mnist_cnn.keras 없음 → 빠른 학습 시작 (~2분)...')
    import subprocess, sys
    loader = os.path.join(ROOT, 'loaders', 'load_cnn.py')
    subprocess.run([sys.executable, loader], check=True)
    return keras.models.load_model(path)

def load_ddpm_mnist():
    """HuggingFace DDPM MNIST 로드 → HFUNetWrapper 반환"""
    local = os.path.join(MODELS, 'ddpm-mnist')
    if not os.path.exists(local):
        raise FileNotFoundError(
            f'HF DDPM 모델 없음: {local}\n'
            '  먼저 실행: uv run python loaders/load_diffusion.py'
        )
    return HFUNetWrapper(local)


class HFUNetWrapper:
    """
    HuggingFace DDPMPipeline UNet을 Keras 스타일로 래핑.

    viz/06_diffusion.py와 동일한 인터페이스:
        pred = unet([xt, t], training=False)

    - xt: (N, 28, 28, 1) float32, 범위 [-1, 1]
    - t : (N,) int32, 타임스텝 0 ~ T-1
    - 반환: (N, 28, 28, 1) float32 예측 노이즈
    """

    def __init__(self, model_path: str):
        import torch
        from diffusers import DDPMPipeline
        self._torch = torch
        pipe = DDPMPipeline.from_pretrained(model_path)
        self._unet = pipe.unet.eval()

    def __call__(self, inputs, training=False):
        torch = self._torch
        xt, t = inputs

        # TF tensor / numpy 양쪽 처리
        xt_np = xt.numpy() if hasattr(xt, 'numpy') else np.asarray(xt)
        t_np  = t.numpy()  if hasattr(t,  'numpy') else np.asarray(t)

        # NHWC → NCHW
        xt_pt = torch.tensor(xt_np).permute(0, 3, 1, 2).float()
        t_pt  = torch.tensor(t_np.astype('int64')).long()

        with torch.no_grad():
            noise_pred = self._unet(xt_pt, t_pt).sample  # (N, C, H, W)

        # NCHW → NHWC
        return noise_pred.permute(0, 2, 3, 1).cpu().numpy()
