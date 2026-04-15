"""
DCGAN (MNIST) 빠른 학습 — 5 에폭, 1만 샘플 (~2분)
Usage: uv run python loaders/load_gan.py
"""
import sys, os
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
sys.path.insert(0, ROOT)
os.makedirs(MODELS, exist_ok=True)

import numpy as np
import keras
import tensorflow as tf
from arch import build_generator, build_discriminator, LATENT_DIM

# ─── 데이터 (10k 샘플로 축소) ─────────────────────────────────────────────────
(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train[:10000].astype('float32') / 127.5 - 1.0, -1)
print(f'학습 데이터: {x_train.shape}')

# ─── 모델 ─────────────────────────────────────────────────────────────────────
generator     = build_generator()
discriminator = build_discriminator()

cross_entropy = keras.losses.BinaryCrossentropy()
gen_opt       = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
disc_opt      = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

@tf.function
def train_step(real_batch):
    bs    = tf.shape(real_batch)[0]
    noise = tf.random.normal([bs, LATENT_DIM])
    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake     = generator(noise, training=True)
        real_out = discriminator(real_batch, training=True)
        fake_out = discriminator(fake, training=True)
        g_loss   = cross_entropy(tf.ones_like(fake_out) * 0.9, fake_out)
        d_loss   = (cross_entropy(tf.ones_like(real_out) * 0.9, real_out)
                  + cross_entropy(tf.zeros_like(fake_out) + 0.1, fake_out))
    gen_opt.apply_gradients(zip(gt.gradient(g_loss, generator.trainable_variables),
                                generator.trainable_variables))
    disc_opt.apply_gradients(zip(dt.gradient(d_loss, discriminator.trainable_variables),
                                 discriminator.trainable_variables))
    return g_loss, d_loss

# ─── 학습 ─────────────────────────────────────────────────────────────────────
EPOCHS     = 5
BATCH_SIZE = 256

dataset = (tf.data.Dataset.from_tensor_slices(x_train)
           .shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
           .prefetch(tf.data.AUTOTUNE))

fixed_noise = tf.random.normal([16, LATENT_DIM], seed=42)
gen_losses, disc_losses, snapshots = [], [], {}

for epoch in range(EPOCHS):
    g_sum = d_sum = n = 0
    for batch in dataset:
        g, d = train_step(batch)
        g_sum += g.numpy(); d_sum += d.numpy(); n += 1
    gen_losses.append(g_sum / n); disc_losses.append(d_sum / n)
    snapshots[epoch] = generator(fixed_noise, training=False).numpy()
    print(f'Epoch {epoch+1:02d}/{EPOCHS}  Gen={gen_losses[-1]:.4f}  Disc={disc_losses[-1]:.4f}')

# ─── 저장 ─────────────────────────────────────────────────────────────────────
generator.save_weights(os.path.join(MODELS, 'generator.weights.h5'))
discriminator.save_weights(os.path.join(MODELS, 'discriminator.weights.h5'))
np.save(os.path.join(MODELS, 'gan_snapshots.npy'), snapshots, allow_pickle=True)
np.save(os.path.join(MODELS, 'gan_losses.npy'),
        {'gen': gen_losses, 'disc': disc_losses}, allow_pickle=True)
print('저장 완료: models/generator.weights.h5 외')
