"""Gradient 기반 Attribution / Usage: uv run python viz/02_gradient_attribution.py"""
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
from scipy.ndimage import zoom

(_, _), (_, y_test) = keras.datasets.mnist.load_data()
x_test = np.expand_dims(
    keras.datasets.mnist.load_data()[1][0].astype('float32') / 255, -1)

model = keras.models.load_model(os.path.join(MODELS, 'mnist_cnn.keras'))
print('모델 로드 완료')

sample_indices = [np.where(y_test == d)[0][0] for d in range(8)]
n = len(sample_indices)

def pred(img): return np.argmax(model.predict(img[np.newaxis], verbose=0))

# ─── Attribution 함수들 ────────────────────────────────────────────────────────

def vanilla_saliency(img, cls):
    t = tf.constant(img[np.newaxis].astype('float32'))
    with tf.GradientTape() as tape:
        tape.watch(t)
        out = model(t, training=False)
    return tf.abs(tape.gradient(out[:, cls], t)).numpy()[0, :, :, 0]

def guided_backprop(img, cls):
    t = tf.constant(img[np.newaxis].astype('float32'))
    with tf.GradientTape() as tape:
        tape.watch(t)
        out = model(t, training=False)
    g = tape.gradient(out[:, cls], t)
    return (g * tf.cast(g > 0, tf.float32) * tf.cast(t > 0, tf.float32)).numpy()[0, :, :, 0]

def grad_cam(img, cls, layer='conv2d_2'):
    # conv_out은 Variable로 만들어야 tape이 자동 watch함
    feat_model = keras.Model(model.input, model.get_layer(layer).output)

    feat_shape = model.get_layer(layer).output.shape[1:]
    sub_in = keras.Input(shape=feat_shape)
    x = sub_in
    past = False
    for lyr in model.layers:
        if past:
            x = lyr(x)
        elif lyr.name == layer:
            past = True
    sub_model = keras.Model(sub_in, x)

    img_tensor = tf.constant(img[np.newaxis].astype('float32'))
    conv_out   = tf.Variable(feat_model(img_tensor, training=False))

    with tf.GradientTape() as tape:
        preds = sub_model(conv_out, training=False)
    grads  = tape.gradient(preds[:, cls], conv_out)
    cam    = tf.nn.relu(tf.reduce_sum(conv_out[0] * tf.reduce_mean(grads, (0,1,2)), -1)).numpy()
    h, w   = img.shape[:2]
    cam_up = zoom(cam, (h / cam.shape[0], w / cam.shape[1]))
    return (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-8)

def integrated_gradients(img, cls, steps=50):
    baseline = np.zeros_like(img)
    alphas   = np.linspace(0.0, 1.0, steps + 1)
    interps  = np.array([baseline + a * (img - baseline) for a in alphas], dtype='float32')
    grads    = []
    for x in interps:
        t = tf.cast(x[np.newaxis], tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(t)
            out = model(t, training=False)
        grads.append(tape.gradient(out[:, cls], t).numpy()[0])
    avg = np.mean((np.array(grads[:-1]) + np.array(grads[1:])) / 2, axis=0)
    return ((img - baseline) * avg)[:, :, 0]

# ─── 공통 플롯 함수 ────────────────────────────────────────────────────────────

def save_grid(rows_data, row_labels, title, fname):
    """rows_data: list of (list_of_arrays, cmap)"""
    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, n, figsize=(2.2 * n, 2.5 * n_rows))
    for ri, (maps, cmap) in enumerate(rows_data):
        for ci, (idx, m) in enumerate(zip(sample_indices, maps)):
            axes[ri, ci].imshow(m, cmap=cmap); axes[ri, ci].axis('off')
            if ri == 0:
                p = pred(x_test[idx])
                axes[ri, ci].set_title(f'{y_test[idx]}→{p}', fontsize=9)
        axes[ri, 0].set_ylabel(row_labels[ri], fontsize=9)
    plt.suptitle(title, fontsize=13); plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=100, bbox_inches='tight')
    plt.close(); print(f'저장: {fname}')

# ─── 각 기법 실행 ──────────────────────────────────────────────────────────────

originals = [x_test[i, :, :, 0] for i in sample_indices]

print('Vanilla Saliency...')
sals = [vanilla_saliency(x_test[i], pred(x_test[i])) for i in sample_indices]
save_grid([(originals, 'gray'), (sals, 'hot'),
           ([o*0.4 + s*0.6 for o,s in zip(originals,sals)], 'hot')],
          ['원본', 'Saliency', 'Overlay'], 'Vanilla Saliency Maps', '02_saliency.png')

print('Guided Backprop...')
gbps = [guided_backprop(x_test[i], pred(x_test[i])) for i in sample_indices]
save_grid([(originals, 'gray'), (gbps, 'hot'),
           ([o*0.4 + g*0.6 for o,g in zip(originals,gbps)], 'hot')],
          ['원본', 'Guided BP', 'Overlay'], 'Guided Backpropagation', '02_guided_backprop.png')

print('Grad-CAM...')
cams = [grad_cam(x_test[i], pred(x_test[i])) for i in sample_indices]
save_grid([(originals, 'gray'), (cams, 'jet'),
           ([o*0.5 + c*0.5 for o,c in zip(originals,cams)], 'jet')],
          ['원본', 'Grad-CAM', 'Overlay'], 'Grad-CAM', '02_gradcam.png')

print('Integrated Gradients...')
igs = [integrated_gradients(x_test[i], pred(x_test[i])) for i in sample_indices]
save_grid([(originals, 'gray'), (igs, 'RdBu_r'), ([np.abs(ig) for ig in igs], 'hot')],
          ['원본', 'IG (R=+, B=-)', '|IG| Overlay'],
          'Integrated Gradients', '02_integrated_gradients.png')

# ─── 비교 (단일 이미지) ────────────────────────────────────────────────────────
idx  = sample_indices[3]
img  = x_test[idx]
p    = pred(img)
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, (data, title, cmap) in zip(axes, [
    (img[:,:,0],                         '원본',              'gray'),
    (vanilla_saliency(img, p),           'Vanilla Saliency', 'hot'),
    (guided_backprop(img, p),            'Guided Backprop',  'hot'),
    (grad_cam(img, p),                   'Grad-CAM',         'jet'),
    (np.abs(integrated_gradients(img,p)),'Integrated Grads', 'hot'),
]):
    ax.imshow(data, cmap=cmap); ax.set_title(title, fontsize=11); ax.axis('off')
plt.suptitle(f'Attribution 비교 — 숫자 {y_test[idx]} (예측: {p})', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '02_comparison.png'), dpi=120, bbox_inches='tight')
plt.close(); print('저장: 02_comparison.png')

print('완료!')
