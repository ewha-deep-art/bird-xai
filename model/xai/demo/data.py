"""
데이터셋 설정 — DATASET 한 줄만 바꾸면 전체 파이프라인이 변경됩니다.

지원 데이터셋:
  'mnist'         — 28×28 grayscale, 10 classes (0-9)
  'fashion_mnist' — 28×28 grayscale, 10 classes (의류)
  'cifar10'       — 32×32 RGB,       10 classes (자동차, 동물 등)
"""
import numpy as np
import keras

# ─── 여기만 바꾸세요 ───────────────────────────────────────────────────────────
DATASET = 'mnist'
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY = {
    'mnist':         keras.datasets.mnist,
    'fashion_mnist': keras.datasets.fashion_mnist,
    'cifar10':       keras.datasets.cifar10,
}
_CLASS_NAMES = {
    'mnist':         [str(i) for i in range(10)],
    'fashion_mnist': ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    'cifar10':       ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck'],
}


def load_dataset(name=None):
    """
    Returns:
        (x_train, y_train), (x_test, y_test), input_shape, num_classes, class_names

    - x: float32, [0, 1], shape (N, H, W, C)
    - y: int (flat), shape (N,)
    """
    if name is None:
        name = DATASET
    if name not in _REGISTRY:
        raise ValueError(f"지원하지 않는 데이터셋: '{name}'. 지원: {list(_REGISTRY)}")

    (x_train, y_train), (x_test, y_test) = _REGISTRY[name].load_data()
    y_train, y_test = y_train.ravel(), y_test.ravel()

    # 채널 추가 (grayscale인 경우)
    if x_train.ndim == 3:
        x_train = np.expand_dims(x_train, -1)
        x_test  = np.expand_dims(x_test,  -1)

    x_train = x_train.astype('float32') / 255
    x_test  = x_test.astype('float32') / 255

    input_shape = x_train.shape[1:]   # (H, W, C)
    num_classes  = len(_CLASS_NAMES[name])
    class_names  = _CLASS_NAMES[name]

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes, class_names


def load_dataset_gan(name=None):
    """GAN용: [-1, 1] 범위로 정규화"""
    (x_train, y_train), (x_test, y_test), input_shape, num_classes, class_names = load_dataset(name)
    return (x_train * 2 - 1, y_train), (x_test * 2 - 1, y_test), input_shape, num_classes, class_names


# ─── 시각화 유틸 ───────────────────────────────────────────────────────────────

def to_display(img):
    """matplotlib 표시용: grayscale이면 채널 차원 제거, RGB는 그대로"""
    return img[:, :, 0] if img.shape[-1] == 1 else img

def display_cmap(input_shape):
    """grayscale이면 'gray', RGB이면 None"""
    return 'gray' if input_shape[-1] == 1 else None
