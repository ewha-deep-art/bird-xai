"""
MNIST CNN 학습 및 저장 (CPU 최적화)
Usage: uv run python loaders/load_cnn.py

arch.py의 build_cnn()으로 MNIST CNN을 학습합니다.
GPU 없이도 ~2분 내로 완료됩니다.
"""
import sys, os
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
sys.path.insert(0, ROOT)
os.makedirs(MODELS, exist_ok=True)

import numpy as np
import keras
from keras.utils import to_categorical
from arch import build_cnn

save_path = os.path.join(MODELS, 'mnist_cnn.keras')
if os.path.exists(save_path):
    print(f'이미 존재: {save_path}')
    sys.exit(0)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train.astype('float32') / 255, -1)
x_test  = np.expand_dims(x_test.astype('float32') / 255, -1)

model = build_cnn()
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, to_categorical(y_train, 10),
          batch_size=256, epochs=5, validation_split=0.1,
          verbose=1)

loss, acc = model.evaluate(x_test, to_categorical(y_test, 10), verbose=0)
print(f'Test accuracy: {acc:.4f}')

model.save(save_path)
print(f'저장 완료: {save_path}')
