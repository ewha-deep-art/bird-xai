"""SHAP (DeepExplainer / GradientExplainer) / Usage: uv run python viz/07_shap.py"""
import sys, os
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
OUT    = os.path.join(ROOT, 'outputs')
sys.path.insert(0, ROOT)
os.makedirs(OUT, exist_ok=True)

import numpy as np
import keras
import shap
import matplotlib.pyplot as plt

(_, _), (x_test_raw, y_test) = keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test_raw.astype('float32') / 255, -1)

model = keras.models.load_model(os.path.join(MODELS, 'mnist_cnn.keras'))
print('모델 로드 완료')

# 배경 샘플 (DeepExplainer용, 클래스별 10개씩)
rng = np.random.default_rng(42)
bg_indices = np.concatenate([
    rng.choice(np.where(y_test == c)[0], size=10, replace=False)
    for c in range(10)
])
background = x_test[bg_indices]   # (100, 28, 28, 1)

# 설명할 샘플: 숫자 0~9 각 1개
sample_indices = [np.where(y_test == d)[0][0] for d in range(10)]
x_samples = x_test[sample_indices]   # (10, 28, 28, 1)

# ─── 1. DeepExplainer ─────────────────────────────────────────────────────────

print('DeepExplainer 계산 중...')
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(x_samples)   # (10, 10, 28, 28, 1)

# 각 샘플에 대해 예측 클래스의 SHAP 맵을 추출
preds = np.argmax(model.predict(x_samples, verbose=0), axis=1)
shap_pred = np.array([shap_values[preds[i]][i, :, :, 0] for i in range(10)])

# 원본 / SHAP / Overlay 3행 그리드
fig, axes = plt.subplots(3, 10, figsize=(22, 7))
for i in range(10):
    img  = x_samples[i, :, :, 0]
    smap = shap_pred[i]
    vmax = np.abs(smap).max()

    axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f'{y_test[sample_indices[i]]}→{preds[i]}', fontsize=9)

    axes[1, i].imshow(smap, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    axes[2, i].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[2, i].imshow(smap, cmap='RdBu_r', alpha=0.6, vmin=-vmax, vmax=vmax)

    for ax in axes[:, i]:
        ax.axis('off')

axes[0, 0].set_ylabel('원본',               fontsize=10)
axes[1, 0].set_ylabel('SHAP\n(Red=+, Blue=-)', fontsize=9)
axes[2, 0].set_ylabel('Overlay',            fontsize=10)
plt.suptitle('SHAP DeepExplainer — 예측 클래스 기여도', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '07_shap_deep.png'), dpi=100, bbox_inches='tight')
plt.close()
print('저장: 07_shap_deep.png')

# ─── 2. 클래스별 SHAP 비교 (숫자 하나 → 10개 클래스 전부) ─────────────────────

print('클래스별 SHAP 계산 중...')
idx  = sample_indices[8]   # 숫자 8
img  = x_test[idx]
pred = np.argmax(model.predict(img[np.newaxis], verbose=0))
sv_single = explainer.shap_values(img[np.newaxis])  # list[10] of (1,28,28,1)

fig, axes = plt.subplots(2, 10, figsize=(22, 5))
for c in range(10):
    smap = sv_single[c][0, :, :, 0]
    vmax = np.abs(smap).max()
    axes[0, c].imshow(img[:, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[0, c].set_title(f'class {c}', fontsize=9)
    axes[0, c].axis('off')
    axes[1, c].imshow(smap, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, c].set_xlabel(f'{"★ 예측" if c == pred else ""}', fontsize=8)
    axes[1, c].axis('off')

axes[0, 0].set_ylabel('원본', fontsize=10)
axes[1, 0].set_ylabel('SHAP', fontsize=10)
plt.suptitle(f'숫자 {y_test[idx]} (예측: {pred}) — 10개 클래스에 대한 SHAP', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '07_shap_all_classes.png'), dpi=100, bbox_inches='tight')
plt.close()
print('저장: 07_shap_all_classes.png')

# ─── 3. GradientExplainer ─────────────────────────────────────────────────────

print('GradientExplainer 계산 중...')
g_explainer  = shap.GradientExplainer(model, background)
g_shap_vals  = g_explainer.shap_values(x_samples)   # list[10] of (10,28,28,1)
g_shap_pred  = np.array([g_shap_vals[preds[i]][i, :, :, 0] for i in range(10)])

# DeepExplainer vs GradientExplainer 나란히 비교
fig, axes = plt.subplots(3, 10, figsize=(22, 7))
for i in range(10):
    img  = x_samples[i, :, :, 0]
    d_s  = shap_pred[i]
    g_s  = g_shap_pred[i]
    vmax = max(np.abs(d_s).max(), np.abs(g_s).max())

    axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f'{y_test[sample_indices[i]]}→{preds[i]}', fontsize=9)
    axes[1, i].imshow(d_s, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2, i].imshow(g_s, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    for ax in axes[:, i]:
        ax.axis('off')

axes[0, 0].set_ylabel('원본',              fontsize=10)
axes[1, 0].set_ylabel('Deep\nExplainer',   fontsize=9)
axes[2, 0].set_ylabel('Gradient\nExplainer', fontsize=9)
plt.suptitle('SHAP: DeepExplainer vs GradientExplainer', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '07_shap_comparison.png'), dpi=100, bbox_inches='tight')
plt.close()
print('저장: 07_shap_comparison.png')

print('완료!')
