"""공유 모델 아키텍처 정의 — 학습/시각화 스크립트에서 import해 사용"""
import numpy as np
import tensorflow as tf
import keras
from keras import layers

# ─── MNIST CNN ────────────────────────────────────────────────────────────────

def build_cnn():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1'),
        layers.MaxPooling2D((2, 2), name='maxpool_1'),
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        layers.MaxPooling2D((2, 2), name='maxpool_2'),
        layers.Flatten(name='flatten'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ], name='mnist_cnn')


# ─── DCGAN ────────────────────────────────────────────────────────────────────

LATENT_DIM = 100

def build_generator(latent_dim=LATENT_DIM):
    return keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(7 * 7 * 256),
        layers.Reshape((7, 7, 256)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2D(1, (3, 3), padding='same', activation='tanh'),
    ], name='generator')

def build_discriminator():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', name='disc_conv1'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', name='disc_conv2'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid'),
    ], name='discriminator')


# ─── DDPM ─────────────────────────────────────────────────────────────────────

T          = 1000
BETA_START = 1e-4
BETA_END   = 2e-2

def get_noise_schedule():
    betas     = np.linspace(BETA_START, BETA_END, T).astype('float32')
    alphas    = (1.0 - betas).astype('float32')
    alpha_bar = np.cumprod(alphas).astype('float32')
    return betas, alphas, alpha_bar


class SinusoidalEmbedding(keras.layers.Layer):
    """타임스텝 t → sinusoidal 임베딩"""
    def __init__(self, dim=32, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, t):
        half   = self.dim // 2
        freqs  = tf.exp(-tf.math.log(10000.0) * tf.cast(tf.range(half), tf.float32) / half)
        t_f    = tf.cast(t, tf.float32)
        args   = t_f[:, tf.newaxis] * freqs[tf.newaxis, :]
        return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)

    def get_config(self):
        return {**super().get_config(), 'dim': self.dim}


def build_unet(t_emb_dim=32):
    """MNIST용 경량 U-Net (각 타임스텝에서 노이즈 예측)"""
    img_in = keras.Input(shape=(28, 28, 1), name='img_input')
    t_in   = keras.Input(shape=(), dtype=tf.int32, name='t_input')

    # Time embedding
    t_emb = SinusoidalEmbedding(t_emb_dim)(t_in)           # (B, t_emb_dim)
    t_emb = layers.Dense(64, activation='swish')(t_emb)
    t_emb = layers.Dense(64, activation='swish')(t_emb)

    def time_add(x):
        """feature map에 time embedding을 채널별로 더함"""
        ch     = x.shape[-1]
        t_proj = layers.Dense(ch)(t_emb)                   # (B, ch)
        t_proj = layers.Reshape((1, 1, ch))(t_proj)        # (B, 1, 1, ch)
        return x + t_proj

    # Encoder
    x1 = layers.Conv2D(32, 3, padding='same', activation='swish')(img_in)  # 28×28
    x1 = time_add(x1)
    x1 = layers.Conv2D(32, 3, padding='same', activation='swish')(x1)

    x2 = layers.Conv2D(64, 3, strides=2, padding='same', activation='swish')(x1)  # 14×14
    x2 = time_add(x2)
    x2 = layers.Conv2D(64, 3, padding='same', activation='swish')(x2)

    x3 = layers.Conv2D(128, 3, strides=2, padding='same', activation='swish')(x2)  # 7×7
    x3 = time_add(x3)
    x3 = layers.Conv2D(128, 3, padding='same', activation='swish')(x3)

    # Bottleneck
    xb = layers.Conv2D(256, 3, padding='same', activation='swish')(x3)
    xb = time_add(xb)
    xb = layers.Conv2D(128, 3, padding='same', activation='swish')(xb)

    # Decoder
    xd2 = layers.UpSampling2D(2)(xb)
    xd2 = layers.Concatenate()([xd2, x2])
    xd2 = layers.Conv2D(64, 3, padding='same', activation='swish')(xd2)
    xd2 = time_add(xd2)

    xd1 = layers.UpSampling2D(2)(xd2)
    xd1 = layers.Concatenate()([xd1, x1])
    xd1 = layers.Conv2D(32, 3, padding='same', activation='swish')(xd1)
    xd1 = time_add(xd1)

    out = layers.Conv2D(1, 1)(xd1)  # 예측 노이즈

    return keras.Model([img_in, t_in], out, name='unet')
