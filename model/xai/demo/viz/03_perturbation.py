"""Occlusion Sensitivity + LIME / Usage: uv run python viz/03_perturbation.py"""
import sys, os
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
OUT    = os.path.join(ROOT, 'outputs')
sys.path.insert(0, ROOT)
os.makedirs(OUT, exist_ok=True)

import numpy as np
import keras
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

(_, _), (_, y_test) = keras.datasets.mnist.load_data()
x_test = np.expand_dims(
    keras.datasets.mnist.load_data()[1][0].astype('float32') / 255, -1)

model = keras.models.load_model(os.path.join(MODELS, 'mnist_cnn.keras'))
print('모델 로드 완료')

sample_indices = [np.where(y_test == d)[0][0] for d in range(8)]
n = len(sample_indices)
def pred(img): return np.argmax(model.predict(img[np.newaxis], verbose=0))

# ─── 1. Occlusion Sensitivity ─────────────────────────────────────────────────

def occlusion(img, cls, ps=4, stride=2):
    h, w    = img.shape[:2]
    base    = model.predict(img[np.newaxis], verbose=0)[0, cls]
    scores  = np.zeros((h, w)); counts = np.zeros((h, w))
    for r in range(0, h - ps + 1, stride):
        for c in range(0, w - ps + 1, stride):
            occ = img.copy(); occ[r:r+ps, c:c+ps, :] = 0
            sc  = model.predict(occ[np.newaxis], verbose=0)[0, cls]
            scores[r:r+ps, c:c+ps] += base - sc
            counts[r:r+ps, c:c+ps] += 1
    return scores / np.maximum(counts, 1)

print('Occlusion Sensitivity 계산 중...')
fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 7))
for i, idx in enumerate(sample_indices):
    img = x_test[idx]; p = pred(img)
    occ = occlusion(img, p)
    axes[0, i].imshow(img[:,:,0], cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f'{y_test[idx]}→{p}', fontsize=9)
    axes[1, i].imshow(occ, cmap='RdYlGn')
    axes[2, i].imshow(img[:,:,0], cmap='gray', vmin=0, vmax=1)
    axes[2, i].imshow(np.clip(occ, 0, None), cmap='hot', alpha=0.6)
    for ax in axes[:, i]: ax.axis('off')
    print(f'  [{i+1}/{n}]')
axes[0,0].set_ylabel('원본', fontsize=10)
axes[1,0].set_ylabel('Occlusion\n(Green=중요)', fontsize=9)
axes[2,0].set_ylabel('Overlay', fontsize=10)
plt.suptitle('Occlusion Sensitivity (4×4 패치)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '03_occlusion.png'), dpi=100, bbox_inches='tight')
plt.close(); print('저장: 03_occlusion.png')

# 패치 크기 비교
idx = sample_indices[5]; img = x_test[idx]; p = pred(img)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].imshow(img[:,:,0], cmap='gray', vmin=0, vmax=1)
axes[0].set_title(f'원본 (숫자 {y_test[idx]})', fontsize=12); axes[0].axis('off')
for ax, ps in zip(axes[1:], [2, 4, 7]):
    print(f'  패치 {ps}×{ps}...')
    occ = occlusion(img, p, ps=ps, stride=max(1, ps//2))
    ax.imshow(img[:,:,0], cmap='gray', vmin=0, vmax=1)
    ax.imshow(np.clip(occ, 0, None), cmap='hot', alpha=0.65)
    ax.set_title(f'패치 {ps}×{ps}', fontsize=12); ax.axis('off')
plt.suptitle('Occlusion — 패치 크기 비교', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '03_occlusion_patch_sizes.png'), dpi=100, bbox_inches='tight')
plt.close(); print('저장: 03_occlusion_patch_sizes.png')

# ─── 2. LIME ──────────────────────────────────────────────────────────────────

def predict_lime(imgs_rgb):
    gray = imgs_rgb.mean(axis=-1, keepdims=True).astype('float32')
    return model.predict(gray, verbose=0)

explainer = lime_image.LimeImageExplainer()
print('LIME 계산 중...')
fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 7))
for i, idx in enumerate(sample_indices):
    img = x_test[idx]; p = pred(img)
    img_rgb = np.stack([img[:,:,0]] * 3, axis=-1)
    exp = explainer.explain_instance(
        img_rgb, predict_lime, top_labels=1, hide_color=0,
        num_samples=500, random_seed=42)
    label = exp.top_labels[0]
    pos, pm = exp.get_image_and_mask(label, positive_only=True, num_features=5, hide_rest=False)
    neg, nm = exp.get_image_and_mask(label, positive_only=False, negative_only=True,
                                      num_features=5, hide_rest=False)
    axes[0, i].imshow(img[:,:,0], cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f'{y_test[idx]}→{p}', fontsize=9)
    axes[1, i].imshow(mark_boundaries(pos, pm))
    axes[2, i].imshow(mark_boundaries(neg, nm))
    for ax in axes[:, i]: ax.axis('off')
    print(f'  [{i+1}/{n}]')
axes[0,0].set_ylabel('원본', fontsize=10)
axes[1,0].set_ylabel('LIME (긍정)', fontsize=9)
axes[2,0].set_ylabel('LIME (부정)', fontsize=9)
plt.suptitle('LIME — 슈퍼픽셀 기반 중요도', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '03_lime.png'), dpi=100, bbox_inches='tight')
plt.close(); print('저장: 03_lime.png')

print('완료!')
