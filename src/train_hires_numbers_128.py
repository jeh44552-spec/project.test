# src/train_hires_numbers_128.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import numpy as np
import os

DATASET_PATH = '../hires_numbers'
# --- 해상도를 128x128로 대폭 상향! ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
# -----------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=32, color_mode='grayscale'
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=32, color_mode='grayscale'
)

class_names = train_ds.class_names
print("학습할 숫자 (128x128):", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    Conv2D(32, 3, padding='same', activation='relu'), MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'), MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'), MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=15) # 시간은 더 오래 걸립니다.

if not os.path.exists('../models'): os.makedirs('../models')
# 새 모델 파일 이름 사용
np.save('../models/hires_number_class_names_128.npy', class_names)
model.save('../models/hires_number_model_128.h5')
print("초고해상도 숫자 전문가 모델이 저장되었습니다.")