# src/train_transfer_learning.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2 # 초엘리트 전문가 모델 임포트
import numpy as np
import os

# 1. 현재 우리의 데이터셋 (숫자 + '+', '-')
DATASET_PATH = '../final_dataset' # 숫자와 +-가 들어있는 폴더
IMG_HEIGHT = 128
IMG_WIDTH = 128

# 2. 데이터셋 불러오기 (컬러 모드로 변경!)
def preprocess(image, label):
    # 흑백(1채널) 이미지를 3채널로 복사하는 함수
    image = tf.image.grayscale_to_rgb(image)
    return image, label

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=32,
    color_mode='grayscale' # 일단 흑백으로 불러온 뒤,
).map(preprocess) # map 함수를 이용해 3채널로 변환!

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=32,
    color_mode='grayscale'
).map(preprocess)

class_names = np.array(train_ds.class_names)
print("전이 학습으로 학습할 최종 클래스:", class_names)

if not os.path.exists('../models'): os.makedirs('../models')
np.save('../models/transfer_class_names.npy', class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 3. 초엘리트 전문가(MobileNetV2) 모델 불러오기
# include_top=False: MobileNetV2의 마지막 분류기는 버리고, 특징 추출 부분만 사용하겠다는 의미
base_model = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                         include_top=False,
                         weights='imagenet') # ImageNet 데이터로 사전 훈련된 가중치 사용

# 전문가의 지식은 그대로 유지하도록 동결(Freeze)
base_model.trainable = False

# 4. 우리의 맞춤형 분류기 추가하기
inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = base_model(inputs, training=False) # 전문가에게 특징 추출 요청
x = GlobalAveragePooling2D()(x) # 분석된 특징들을 깔끔하게 정리
x = Dropout(0.2)(x)
# 마지막 출력층만 우리가 직접 만듦
outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs, outputs)

# 5. 모델 컴파일 및 훈련
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 전이 학습은 훨씬 더 빨리 수렴하므로, 에포크를 줄여도 됨
model.fit(train_ds, validation_data=val_ds, epochs=10)

# 6. 최종 전이 학습 모델 저장
model.save('../models/transfer_model.h5')
print("최종 전이 학습 모델이 저장되었습니다.")