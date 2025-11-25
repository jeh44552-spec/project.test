# src/train_ultimate.py

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from CustomDataGenerator import CustomDataGenerator # 우리가 만든 제너레이터 임포트

# --- 데이터셋 설정 ---
TRAIN_DIR = '../final_dataset' # 여기서는 하나의 폴더만 필요
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# 1. 커스텀 데이터 제너레이터 생성
# 참고: 이 방식으로는 validation_split이 복잡하므로, 우선 전체 데이터로 훈련합니다.
train_generator = CustomDataGenerator(TRAIN_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
class_names = np.array(train_generator.class_names)
print("최종 학습 클래스:", class_names)

if not os.path.exists('../models'): os.makedirs('../models')
np.save('../models/ultimate_class_names.npy', class_names)

# --- 미세 조정 모델 구성 (이전과 동일) ---
base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                         include_top=False, weights='imagenet')
base_model.trainable = False 

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--- 1단계: 상위 분류기 훈련 시작 ---")
model.fit(train_generator, epochs=10)

print("\n--- 2단계: 미세 조정 시작 ---")
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=['accuracy'])

model.fit(train_generator, epochs=10) # 추가로 10 에포크 더 미세 조정

# --- 최종 모델 저장 ---
model.save('../models/ultimate_model.h5')
print("궁극의 최종 모델이 저장되었습니다.")