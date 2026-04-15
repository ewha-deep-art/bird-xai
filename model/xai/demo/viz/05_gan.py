"""GAN 시각화 / Usage: uv run python viz/05_gan.py"""
import sys, os
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
OUT    = os.path.join(ROOT, 'outputs')
sys.path.insert(0, ROOT)
os.makedirs(OUT, exist_ok=True)

import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from arch import build_generator, build_discriminator, LATENT_DIM

generator     = build_generator()
discriminator = build_discriminator()
generator(tf.zeros([1, LATENT_DIM])); discriminator(tf.zeros([1, 28, 28, 1]))
generator.load_weights(os.path.join(MODELS, 'generator.weights.h5'))
discriminator.load_weights(os.path.join(MODELS, 'discriminator.weights.h5'))
print('모델 로드 완료')

snapshots = np.load(os.path.join(MODELS, 'gan_snapshots.npy'), allow_pickle=True).item()
losses    = np.load(os.path.join(MODELS, 'gan_losses.npy'),    allow_pickle=True).item()
EPOCHS    = len(losses['gen'])

def save(fname): plt.savefig(os.path.join(OUT, fname), dpi=100, bbox_inches='tight'); plt.close(); print(f'저장: {fname}')
def saveb(fname): plt.savefig(os.path.join(OUT, fname), dpi=100, bbox_inches='tight', facecolor='black'); plt.close(); print(f'저장: {fname}')

# ─── 1. Loss 곡선 ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, EPOCHS+1), losses['gen'],  label='Generator',     color='steelblue', lw=2)
ax.plot(range(1, EPOCHS+1), losses['disc'], label='Discriminator', color='tomato',    lw=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('DCGAN Training Loss'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); save('05_loss_curve.png')

# ─── 2. 학습 진행 시각화 ──────────────────────────────────────────────────────
all_ep = sorted(snapshots.keys())
show   = all_ep[::max(1, EPOCHS // 6)]
n_cols = 8
fig, axes = plt.subplots(len(show), n_cols,
                          figsize=(n_cols * 1.5, len(show) * 1.8), facecolor='black')
for row, ep in enumerate(show):
    imgs = snapshots[ep]
    for col in range(n_cols):
        axes[row, col].imshow((imgs[col,:,:,0] + 1) / 2, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off'); axes[row, col].set_facecolor('black')
    axes[row, 0].set_ylabel(f'Epoch {ep+1}', fontsize=10, color='white')
plt.suptitle('DCGAN — 에폭별 생성 이미지 (고정 노이즈)', fontsize=13, color='white')
plt.tight_layout(); saveb('05_training_progression.png')

# ─── 3. 최종 생성 샘플 ────────────────────────────────────────────────────────
final = generator(tf.random.normal([64, LATENT_DIM], seed=7), training=False).numpy()
fig, axes = plt.subplots(8, 8, figsize=(12, 12), facecolor='black')
for i, ax in enumerate(axes.flat):
    ax.imshow((final[i,:,:,0] + 1) / 2, cmap='gray', vmin=0, vmax=1); ax.axis('off')
plt.suptitle('DCGAN 최종 생성 이미지 64개', fontsize=13, color='white')
plt.tight_layout(); saveb('05_final_samples.png')

# ─── 4. Latent Space 보간 ─────────────────────────────────────────────────────
N_INTERP = 14; N_PAIRS = 5
np.random.seed(42)
fig, axes = plt.subplots(N_PAIRS, N_INTERP,
                          figsize=(N_INTERP * 1.6, N_PAIRS * 2), facecolor='black')
for row in range(N_PAIRS):
    z1 = np.random.randn(LATENT_DIM).astype('float32')
    z2 = np.random.randn(LATENT_DIM).astype('float32')
    zs = np.array([z1 + (z2 - z1) * t for t in np.linspace(0, 1, N_INTERP)])
    imgs = generator(zs, training=False).numpy()
    for col in range(N_INTERP):
        axes[row, col].imshow((imgs[col,:,:,0] + 1) / 2, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title(f'{col/(N_INTERP-1):.2f}', fontsize=7, color='white')
    axes[row, 0].set_ylabel(f'쌍 {row+1}', fontsize=9, color='white')
plt.suptitle('Latent Space 선형 보간 (z₁ → z₂)', fontsize=13, color='white')
plt.tight_layout(); saveb('05_latent_interpolation.png')

# ─── 5. Discriminator Feature Map ─────────────────────────────────────────────
disc_feat = keras.Model(discriminator.inputs, discriminator.get_layer('disc_conv2').output)

(x_train_raw, _), _ = keras.datasets.mnist.load_data()
x_real = np.expand_dims(x_train_raw[:4].astype('float32') / 127.5 - 1.0, -1)
x_fake = generator(tf.random.normal([4, LATENT_DIM]), training=False).numpy()
rf = disc_feat.predict(x_real, verbose=0)
ff = disc_feat.predict(x_fake, verbose=0)

N_FILT = 6
fig, axes = plt.subplots(8, N_FILT + 1, figsize=((N_FILT+1)*2, 16))
for i in range(4):
    for row, (img_data, feats, tag) in enumerate([(x_real, rf, 'Real'), (x_fake, ff, 'Fake')]):
        r = i * 2 + row
        axes[r, 0].imshow((img_data[i,:,:,0] + 1)/2, cmap='gray', vmin=0, vmax=1)
        axes[r, 0].set_title(tag, fontsize=8); axes[r, 0].axis('off')
        for j in range(N_FILT):
            axes[r, j+1].imshow(feats[i,:,:,j], cmap='viridis')
            axes[r, j+1].set_title(f'F{j}', fontsize=7); axes[r, j+1].axis('off')
plt.suptitle('Discriminator Feature Maps — 진짜 vs 가짜 (disc_conv2)', fontsize=12)
plt.tight_layout(); save('05_discriminator_features.png')

print('완료!')
