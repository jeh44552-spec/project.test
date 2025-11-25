# src/train_final_model.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import os

# 1. 우리가 직접 만든 "최종 교재" 폴더를 사용합니다.
DATASET_PATH = '../final_dataset'
IMG_HEIGHT = 128
IMG_WIDTH = 128

# 2. 데이터셋 불러오기
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=32, color_mode='grayscale'
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=32, color_mode='grayscale'
)

class_names = train_ds.class_names
print("마스터 전문가가 학습할 최종 클래스:", class_names)
print(f"총 클래스 개수: {len(class_names)}개")

# 3. 모델과 클래스 이름 저장 준비
if not os.path.exists('../models'): os.makedirs('../models')
np.save('../models/final_class_names.npy', class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. 최종 마스터 모델 구축
model = Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    Conv2D(32, 3, padding='same', activation='relu'), MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'), MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'), MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

# 5. 모델 컴파일 및 훈련 (더 안정적인 학습을 위해 에포크 조정)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=15)

# 6. 최종 마스터 모델 저장
model.save('../models/final_model.h5')
print("최종 마스터 전문가 모델이 저장되었습니다.")