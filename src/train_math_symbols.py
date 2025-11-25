# src/train_math_symbols.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import os

# 1. 데이터셋 경로 및 파라미터 설정
DATASET_PATH = '../math_dataset'
IMG_HEIGHT = 45 # 데이터셋 이미지 크기가 45x45임
IMG_WIDTH = 45
BATCH_SIZE = 32

# 2. 폴더에서 이미지 데이터 불러오기 (훈련/검증 데이터로 자동 분리)
# Keras의 이 기능은 폴더 이름을 자동으로 라벨(정답)으로 사용합니다.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale' # 흑백 이미지로 불러오기
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

# 3. 클래스(기호) 이름 저장 (매우 중요!)
# 모델은 0, 1, 2...를 예측하지만, 우리는 이게 '+', '-', 'times'인지 알아야 합니다.
class_names = train_ds.class_names
print("인식할 기호들:", class_names)
print(f"총 클래스 개수: {len(class_names)}개")

# 나중에 predict.py에서 사용하기 위해 클래스 이름을 파일로 저장
if not os.path.exists('../models'):
    os.makedirs('../models')
np.save('../models/math_class_names.npy', class_names)

# 4. 성능 최적화를 위한 데이터 전처리
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 5. 수학 기호 인식을 위한 CNN 모델 구축
num_classes = len(class_names)
model = Sequential([
    # 0~255 값을 0~1 사이로 정규화
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax') # 최종 출력은 클래스 개수만큼!
])

# 6. 모델 컴파일 및 훈련
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # 라벨이 원-핫 인코딩이 아닐 때 사용
              metrics=['accuracy'])

model.summary()

epochs = 15 # 데이터가 많으므로 에포크를 늘려줌
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# 7. 훈련된 모델 저장
model.save('../models/math_symbol_model.h5')
print("수학 기호 인식 모델이 'models/math_symbol_model.h5'에 저장되었습니다.")