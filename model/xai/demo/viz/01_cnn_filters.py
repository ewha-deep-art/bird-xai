"""CNN 필터 & 활성화 시각화 / Usage: uv run python viz/01_cnn_filters.py"""
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

(_, _), (x_test_raw, y_test) = keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test_raw.astype('float32') / 255, -1)

model = keras.models.load_model(os.path.join(MODELS, 'mnist_cnn.keras'))
print('모델 로드 완료')

# ─── 1. Filter Weight Visualization ───────────────────────────────────────────
for layer_name, rows, cols, cmap in [
    ('conv2d_1', 4,  8,  'gray'),
    ('conv2d_2', 8,  8,  'RdBu_r'),
]:
    w = model.get_layer(layer_name).get_weights()[0]   # (3,3,in,out)
    n_filters = w.shape[-1]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        f = w[:, :, 0, i]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f, cmap=cmap, interpolation='nearest')
        ax.axis('off'); ax.set_title(f'F{i}', fontsize=7)
    plt.suptitle(f'{layer_name} — 학습된 필터 {n_filters}개 (3×3)', fontsize=13)
    plt.tight_layout()
    fname = os.path.join(OUT, f'01_filter_weights_{layer_name}.png')
    plt.savefig(fname, dpi=100, bbox_inches='tight'); plt.close()
    print(f'저장: {os.path.basename(fname)}')

# ─── 2. Feature Map Visualization ─────────────────────────────────────────────
layer_names = ['conv2d_1', 'maxpool_1', 'conv2d_2', 'maxpool_2']
act_model   = keras.Model(inputs=model.inputs,
                          outputs=[model.get_layer(n).output for n in layer_names])

for sample_idx in [0, 7, 42]:
    img  = x_test[sample_idx:sample_idx+1]
    pred = np.argmax(model.predict(img, verbose=0))
    acts = act_model.predict(img, verbose=0)

    fig, axes = plt.subplots(len(acts) + 1, 1, figsize=(22, 14), facecolor='#1a1a2e')
    axes[0].imshow(img[0, :, :, 0], cmap='gray')
    axes[0].set_title(f'원본  정답={y_test[sample_idx]}  예측={pred}',
                      fontsize=12, color='white')
    axes[0].axis('off'); axes[0].set_facecolor('#1a1a2e')
    for j, (act, name) in enumerate(zip(acts, layer_names)):
        n_show   = min(act.shape[-1], 32)
        combined = np.hstack([act[0, :, :, k] for k in range(n_show)])
        axes[j+1].imshow(combined, aspect='auto', cmap='viridis')
        axes[j+1].set_title(f'{name}  {act.shape[1:]}', fontsize=11, color='white')
        axes[j+1].axis('off'); axes[j+1].set_facecolor('#1a1a2e')
    plt.suptitle('Feature Map Visualization', fontsize=14, color='white')
    plt.tight_layout()
    fname = os.path.join(OUT, f'01_feature_maps_s{sample_idx}.png')
    plt.savefig(fname, dpi=100, bbox_inches='tight', facecolor='#1a1a2e'); plt.close()
    print(f'저장: {os.path.basename(fname)}')

# ─── 3. Activation Maximization ───────────────────────────────────────────────
def maximize_filter(layer_name, filter_idx, iters=200, lr=2.0):
    feat = keras.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
    img  = tf.Variable(np.random.uniform(0.4, 0.6, (1, 28, 28, 1)).astype('float32'))
    opt  = tf.optimizers.Adam(lr)
    for _ in range(iters):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(feat(img, training=False)[:, :, :, filter_idx])
        opt.apply_gradients([(tape.gradient(loss, img), img)])
        img.assign(tf.clip_by_value(img, 0.0, 1.0))
    return img.numpy()[0, :, :, 0]

print('Activation Maximization 계산 중...')
for layer_name, n_show, rows, cols in [('conv2d_1', 16, 2, 8), ('conv2d_2', 8, 2, 4)]:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    for i, ax in enumerate(axes.flat):
        ax.imshow(maximize_filter(layer_name, i), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Filter {i}', fontsize=9); ax.axis('off')
    plt.suptitle(f'Activation Maximization — {layer_name}', fontsize=13)
    plt.tight_layout()
    fname = os.path.join(OUT, f'01_activation_max_{layer_name}.png')
    plt.savefig(fname, dpi=100, bbox_inches='tight'); plt.close()
    print(f'저장: {os.path.basename(fname)}')

print('완료!')
