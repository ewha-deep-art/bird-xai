"""PCA / t-SNE / UMAP 임베딩 시각화 / Usage: uv run python viz/04_embedding.py"""
import sys, os
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
OUT    = os.path.join(ROOT, 'outputs')
sys.path.insert(0, ROOT)
os.makedirs(OUT, exist_ok=True)

import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

(_, _), (_, y_test) = keras.datasets.mnist.load_data()
x_test = np.expand_dims(
    keras.datasets.mnist.load_data()[1][0].astype('float32') / 255, -1)

model = keras.models.load_model(os.path.join(MODELS, 'mnist_cnn.keras'))
print('모델 로드 완료')

N      = 3000
cmap10 = cm.get_cmap('tab10', 10)

feat_model = keras.Model(inputs=model.inputs, outputs=model.get_layer('flatten').output)
features   = feat_model.predict(x_test[:N], verbose=1)
labels     = y_test[:N]
raw_pix    = x_test[:N].reshape(N, -1)
print(f'Feature shape: {features.shape}')

pca50     = PCA(n_components=50, random_state=42)
feat50    = pca50.fit_transform(features)
raw50     = PCA(n_components=50, random_state=42).fit_transform(raw_pix)

def scatter(ax, result, title):
    for d in range(10):
        mask = labels == d
        ax.scatter(result[mask,0], result[mask,1], c=[cmap10(d)], label=str(d), s=8, alpha=0.65)
    ax.set_title(title, fontsize=11); ax.axis('off')

def save_single(result, title, fname):
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter(ax, result, title)
    ax.legend(title='Digit', markerscale=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=100, bbox_inches='tight')
    plt.close(); print(f'저장: {fname}')

# PCA
pca2_r = PCA(n_components=2, random_state=42).fit_transform(features)
save_single(pca2_r, 'PCA (CNN Feature)', '04_pca.png')

# t-SNE
print('t-SNE (CNN feature) 계산 중...')
tsne_r = TSNE(n_components=2, perplexity=40, random_state=42, verbose=1).fit_transform(feat50)
save_single(tsne_r, 't-SNE: CNN Feature Space', '04_tsne.png')

print('t-SNE (Raw Pixel) 계산 중...')
tsne_raw = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(raw50)
save_single(tsne_raw, 't-SNE: Raw Pixel', '04_tsne_raw.png')

# UMAP
print('UMAP (CNN feature) 계산 중...')
umap_r = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(feat50)
save_single(umap_r, 'UMAP: CNN Feature Space', '04_umap.png')

print('UMAP (Raw Pixel) 계산 중...')
umap_raw = umap.UMAP(n_components=2, random_state=42).fit_transform(raw50)

# 비교 그리드
fig, axes = plt.subplots(2, 3, figsize=(21, 14))
combos = [
    (pca2_r,   'PCA (CNN feature)'),
    (tsne_r,   't-SNE (CNN feature)'),
    (umap_r,   'UMAP (CNN feature)'),
    (PCA(n_components=2).fit_transform(raw_pix), 'PCA (Raw pixel)'),
    (tsne_raw, 't-SNE (Raw pixel)'),
    (umap_raw, 'UMAP (Raw pixel)'),
]
for ax, (r, title) in zip(axes.flat, combos):
    scatter(ax, r, title)
handles, lbls = axes[0,0].get_legend_handles_labels()
fig.legend(handles, lbls, title='Digit', loc='lower center', ncol=10,
           markerscale=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.suptitle('Raw Pixel vs CNN Feature — 임베딩 방법 비교', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '04_comparison.png'), dpi=100, bbox_inches='tight')
plt.close(); print('저장: 04_comparison.png')

print('완료!')
