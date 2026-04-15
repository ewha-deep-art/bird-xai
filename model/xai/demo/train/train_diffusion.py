"""DDPM 학습 및 저장 / Usage: uv run python train/train_diffusion.py"""
import sys, os
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
sys.path.insert(0, ROOT)
os.makedirs(MODELS, exist_ok=True)

import numpy as np
import keras
import tensorflow as tf
from arch import build_unet, get_noise_schedule, T

(x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train.astype('float32') / 127.5 - 1.0, -1)
x_test  = np.expand_dims(x_test.astype('float32') / 127.5 - 1.0, -1)

betas, alphas, alpha_bar = get_noise_schedule()
alpha_bar_tf = tf.constant(alpha_bar)

unet = build_unet()
unet.summary()
print(f'파라미터 수: {unet.count_params():,}')

optimizer = keras.optimizers.Adam(1e-3)

@tf.function
def train_step(x0_batch):
    bs  = tf.shape(x0_batch)[0]
    t   = tf.random.uniform([bs], 0, T, dtype=tf.int32)
    ab  = tf.reshape(tf.gather(alpha_bar_tf, t), [-1, 1, 1, 1])
    eps = tf.random.normal(tf.shape(x0_batch))
    xt  = tf.sqrt(ab) * x0_batch + tf.sqrt(1.0 - ab) * eps
    with tf.GradientTape() as tape:
        pred = unet([xt, t], training=True)
        loss = tf.reduce_mean(tf.square(eps - pred))
    grads = tape.gradient(loss, unet.trainable_variables)
    optimizer.apply_gradients(zip(grads, unet.trainable_variables))
    return loss

EPOCHS     = 3
BATCH_SIZE = 256
dataset = (tf.data.Dataset.from_tensor_slices(x_train)
           .shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
           .prefetch(tf.data.AUTOTUNE))

losses = []
for epoch in range(EPOCHS):
    l_sum = n = 0
    for batch in dataset:
        l_sum += train_step(batch).numpy(); n += 1
    losses.append(l_sum / n)
    print(f'Epoch {epoch+1:02d}/{EPOCHS}  Loss={losses[-1]:.5f}')

unet.save_weights(os.path.join(MODELS, 'unet.weights.h5'))
np.save(os.path.join(MODELS, 'diffusion_losses.npy'), losses)
np.save(os.path.join(MODELS, 'x_test_sample.npy'), x_test[:16])
np.save(os.path.join(MODELS, 'y_test_sample.npy'), y_test[:16])
print('저장: models/unet.weights.h5 / diffusion_losses.npy')
