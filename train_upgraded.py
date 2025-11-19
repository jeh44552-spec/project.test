# src/train_upgraded.py

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# 1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# CNN은 채널(channel) 차원이 필요함 (흑백=1)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. 데이터 증강 (Data Augmentation) 설정
datagen = ImageDataGenerator(
    rotation_range=10,      # 10도 이내로 회전
    width_shift_range=0.1,  # 0.1 비율만큼 좌우 이동
    height_shift_range=0.1, # 0.1 비율만큼 상하 이동
    zoom_range=0.1          # 10% 이내로 확대/축소
)
datagen.fit(x_train)

# 3. 더 강력한 CNN 모델 구축
def build_cnn_model():
    model = Sequential([
        # 입력: (28, 28, 1)
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # 과적합 방지를 위한 드롭아웃
        Dense(10, activation='softmax')
    ])
    return model

model = build_cnn_model()

# 4. 모델 컴파일 및 훈련
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 데이터 증강을 사용하여 모델 훈련
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=10, # 증강 데이터를 사용하므로 에포크를 늘려주는 것이 좋음
          validation_data=(x_test, y_test))

# 5. 훈련된 모델 저장
model_path = '../models/mnist_cnn_augmented_model.h5'
if not os.path.exists('../models'):
    os.makedirs('../models')
model.save(model_path)
print(f"향상된 모델이 '{model_path}'에 저장되었습니다.")