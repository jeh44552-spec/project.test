# src/train_hires_numbers.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import os

DATASET_PATH = '../hires_numbers' # 오직 숫자 데이터만 사용!
IMG_HEIGHT = 45
IMG_WIDTH = 45

# 데이터셋 불러오기
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=32, color_mode='grayscale'
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=32, color_mode='grayscale'
)

class_names = train_ds.class_names
print("학습할 숫자:", class_names)

# 성능 최적화
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 고해상도 숫자 인식을 위한 CNN 모델
num_classes = len(class_names)
model = Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    Conv2D(32, 3, padding='same', activation='relu'), MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'), MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'), MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
# src/train_hires_numbers.py 파일의 맨 아랫부분

# ... model.compile(...) 윗부분은 그대로 ...

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=15)

# --- 이 부분이 가장 중요합니다! ---
if not os.path.exists('../models'): os.makedirs('../models')

# 훈련 시 사용된 실제 클래스 순서를 파일로 저장!
import numpy as np
np.save('../models/hires_number_class_names.npy', class_names)

model.save('../models/hires_number_model.h5')
print("고해상도 숫자 전문가 모델과 '클래스 이름'이 모두 저장되었습니다.")