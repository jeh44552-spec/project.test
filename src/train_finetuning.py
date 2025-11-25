# src/train_finetuning.py

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# --- 데이터셋 설정 (변경 없음) ---
DATASET_PATH = '../final_dataset'
IMG_HEIGHT = 128
IMG_WIDTH = 128

def preprocess(image, label):
    image = tf.image.grayscale_to_rgb(image)
    return image, label

raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=32, color_mode='grayscale'
)
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=32, color_mode='grayscale'
)

class_names = np.array(raw_train_ds.class_names)
print("미세 조정으로 학습할 최종 클래스:", class_names)

train_ds = raw_train_ds.map(preprocess)
val_ds = raw_val_ds.map(preprocess)

if not os.path.exists('../models'): os.makedirs('../models')
np.save('../models/finetuned_class_names.npy', class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 1단계: 전이 학습 (전문가의 지식은 그대로 사용) ---
base_model = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                         include_top=False, weights='imagenet')
base_model.trainable = False # 먼저 전문가를 꽁꽁 얼립니다.

inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=0.001), # 일반적인 학습률로
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--- 1단계: 상위 분류기 훈련 시작 ---")
initial_epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=initial_epochs)

# --- *** 2단계: 미세 조정 (전문가의 지식을 우리 문제에 맞게 살짝 수정) *** ---
print("\n--- 2단계: 미세 조정 시작 ---")
base_model.trainable = True # 이제 전문가의 얼음을 녹입니다.

# MobileNetV2는 총 154개의 층으로 이루어져 있습니다.
# 우리는 하위 층(일반적인 선, 모서리 등)은 계속 얼려두고,
# 상위 층(더 복잡한 패턴)만 녹여서 훈련시킵니다.
fine_tune_at = 100 # 상위 약 54개 층만 훈련

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 미세 조정 시에는 아주 낮은 학습률을 사용해야 합니다!
# 전문가의 지식이 한 번에 망가지는 것을 방지하기 위함입니다.
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-5), # 0.00001
              metrics=['accuracy'])

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

# 이전 훈련에 이어서 계속 훈련합니다.
history_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_ds)
# --------------------------------------------------------------------------

# 최종 미세 조정 모델 저장
model.save('../models/finetuned_model.h5')
print("최종 미세 조정 모델이 저장되었습니다.")