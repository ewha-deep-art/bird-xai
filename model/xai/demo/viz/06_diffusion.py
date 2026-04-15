"""Diffusion 시각화 / Usage: uv run python viz/06_diffusion.py"""
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
from arch import build_unet, get_noise_schedule, T

betas, alphas, alpha_bar = get_noise_schedule()

# ─── 모델 로드: 로컬 Keras 가중치 우선, 없으면 HuggingFace ────────────────────
_unet_path = os.path.join(MODELS, 'unet.weights.h5')
_hf_path   = os.path.join(MODELS, 'ddpm-mnist')

if os.path.exists(_unet_path):
    unet = build_unet()
    unet([tf.zeros([1, 28, 28, 1]), tf.zeros([1], dtype=tf.int32)])
    unet.load_weights(_unet_path)
    print('로컬 Keras U-Net 로드 완료')
elif os.path.exists(_hf_path):
    from model_loader import HFUNetWrapper
    unet = HFUNetWrapper(_hf_path)
    print('HuggingFace DDPM 로드 완료')
else:
    raise FileNotFoundError(
        'Diffusion 모델을 찾을 수 없습니다.\n'
        '  로컬 학습:    uv run python train/train_diffusion.py\n'
        '  HF 다운로드:  uv run python loaders/load_diffusion.py'
    )

# ─── 데이터 & 손실 이력 ───────────────────────────────────────────────────────
(_, _), (x_test_raw, y_test) = keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test_raw.astype('float32') / 127.5 - 1.0, -1)
losses = np.load(os.path.join(MODELS, 'diffusion_losses.npy'))

def save(fname, facecolor=None):
    kw = dict(dpi=100, bbox_inches='tight')
    if facecolor: kw['facecolor'] = facecolor
    plt.savefig(os.path.join(OUT, fname), **kw); plt.close(); print(f'저장: {fname}')

# ─── 1. Noise Schedule ────────────────────────────────────────────────────────
s = 0.008
t_arr  = np.arange(T + 1)
ab_cos = (np.cos((t_arr/T + s)/(1+s) * np.pi/2)**2)[1:]
ab_cos = ab_cos / ab_cos[0]
b_cos  = np.clip(1 - ab_cos / np.concatenate([[1.0], ab_cos[:-1]]), 0, 0.999)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(betas,     label='Linear', color='steelblue')
axes[0].plot(b_cos,     label='Cosine', color='tomato', ls='--')
axes[0].set_title('β Schedule'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(alpha_bar, label='Linear', color='steelblue')
axes[1].plot(ab_cos,    label='Cosine', color='tomato', ls='--')
axes[1].set_title('ᾱ_t (신호 유지 비율)'); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(np.sqrt(alpha_bar),     label='√ᾱ (signal)', color='steelblue')
axes[2].plot(np.sqrt(1-alpha_bar),   label='√(1-ᾱ) (noise)', color='tomato')
axes[2].set_title('신호 vs 노이즈 (Linear)'); axes[2].legend(); axes[2].grid(alpha=0.3)
plt.suptitle('DDPM Noise Schedules', fontsize=13); plt.tight_layout(); save('06_noise_schedule.png')

# ─── 2. Training Loss ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(losses)+1), losses, color='steelblue', lw=2, marker='o')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.set_title('DDPM Training Loss'); ax.grid(alpha=0.3)
plt.tight_layout(); save('06_training_loss.png')

# ─── 3. Forward Diffusion ─────────────────────────────────────────────────────
def fwd(x0, t_idx):
    ab = alpha_bar[t_idx]; eps = np.random.randn(*x0.shape).astype('float32')
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps

sample_imgs = np.array([x_test[np.where(y_test == d)[0][0]] for d in [0,1,7,9]])
t_show = [0, 50, 100, 200, 300, 500, 700, 999]

fig, axes = plt.subplots(len(sample_imgs), len(t_show),
                          figsize=(len(t_show)*1.8, len(sample_imgs)*2), facecolor='black')
for row, x0 in enumerate(sample_imgs):
    for col, t in enumerate(t_show):
        xt  = x0 if t == 0 else fwd(x0, t-1)
        img = np.clip((xt[:,:,0]+1)/2, 0, 1)
        axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title(f't={t}', fontsize=9, color='white')
plt.suptitle('Forward Diffusion q(x_t | x_0)', fontsize=13, color='white')
plt.tight_layout(); save('06_forward_diffusion.png', facecolor='black')

# ─── 4. Reverse Diffusion ─────────────────────────────────────────────────────
def _unet_predict(xt_np, t_val):
    """UNet 노이즈 예측 — Keras / HFUNetWrapper 공통 인터페이스"""
    result = unet([tf.cast(xt_np, tf.float32),
                   tf.constant([t_val]*xt_np.shape[0], dtype=tf.int32)], training=False)
    return result.numpy() if hasattr(result, 'numpy') else np.asarray(result)

def ddpm_step(xt, t_val):
    pred = _unet_predict(xt, t_val)
    b, a, ab = betas[t_val], alphas[t_val], alpha_bar[t_val]
    mean = (1/np.sqrt(a)) * (xt - (b/np.sqrt(1-ab)) * pred)
    if t_val > 0:
        mean += np.sqrt(b) * np.random.randn(*xt.shape).astype('float32')
    return mean

N_SAMP = 8
xt = np.random.randn(N_SAMP, 28, 28, 1).astype('float32')
viz_t = {999, 900, 700, 500, 300, 200, 100, 50, 10, 0}
snaps = {}

print('Reverse diffusion sampling 중...')
for t_val in range(T-1, -1, -1):
    xt = ddpm_step(xt, t_val)
    if t_val in viz_t:
        snaps[t_val] = xt.copy(); print(f'  t={t_val}')
print('완료')

ordered = sorted(viz_t, reverse=True)
fig, axes = plt.subplots(N_SAMP, len(ordered),
                          figsize=(len(ordered)*1.8, N_SAMP*2), facecolor='black')
for col, t in enumerate(ordered):
    for row in range(N_SAMP):
        axes[row, col].imshow(np.clip((snaps[t][row,:,:,0]+1)/2, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
    axes[0, col].set_title(f't={t}', fontsize=9, color='white')
plt.suptitle('Reverse Diffusion — 노이즈에서 이미지 복원', fontsize=13, color='white')
plt.tight_layout(); save('06_reverse_diffusion.png', facecolor='black')

# ─── 5. 노이즈 예측 비교 ──────────────────────────────────────────────────────
test_ts = [50, 200, 500, 800]
x0 = x_test[np.where(y_test == 0)[0][0]]
fig, axes = plt.subplots(4, len(test_ts), figsize=(len(test_ts)*3, 13))
for col, t_val in enumerate(test_ts):
    eps   = np.random.randn(*x0.shape).astype('float32')
    xt_i  = np.sqrt(alpha_bar[t_val]) * x0 + np.sqrt(1-alpha_bar[t_val]) * eps
    pred  = _unet_predict(xt_i[np.newaxis], t_val)[0]
    axes[0, col].imshow(np.clip((x0[:,:,0]+1)/2, 0, 1), cmap='gray')
    axes[0, col].set_title(f't={t_val}', fontsize=11); axes[0, col].axis('off')
    axes[1, col].imshow(np.clip((xt_i[:,:,0]+1)/2, 0, 1), cmap='gray'); axes[1, col].axis('off')
    vmax = np.abs(eps[:,:,0]).max()
    axes[2, col].imshow(eps[:,:,0], cmap='RdBu_r', vmin=-vmax, vmax=vmax); axes[2, col].axis('off')
    vp = np.abs(pred[:,:,0]).max()
    axes[3, col].imshow(pred[:,:,0], cmap='RdBu_r', vmin=-vp, vmax=vp); axes[3, col].axis('off')
for row, lbl in enumerate(['원본 x₀', 'x_t', '실제 ε', '예측 ε_θ']):
    axes[row, 0].set_ylabel(lbl, fontsize=10)
plt.suptitle('타임스텝별 노이즈 예측 비교', fontsize=14)
plt.tight_layout(); save('06_noise_prediction.png')

print('완료!')
