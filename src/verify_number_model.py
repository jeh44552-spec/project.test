# src/verify_number_model.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. 검증할 숫자 모델 로드
try:
    model = tf.keras.models.load_model('../models/hires_number_model.h5')
    print("모델 로드 성공!")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    exit()

# 2. 훈련 때 사용했던 검증 데이터셋 다시 불러오기
DATASET_PATH = '../hires_numbers'
IMG_HEIGHT = 45
IMG_WIDTH = 45

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation", # 검증 데이터만 가져옴
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    color_mode='grayscale'
)

class_names = val_ds.class_names
print("데이터셋의 클래스 순서:", class_names)
print("우리가 예측 코드에서 가정한 순서: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']")

# 3. 데이터셋에서 이미지와 라벨 몇 개 가져오기
plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1): # 1개 배치(32개 이미지)만 테스트
    for i in range(9): # 그 중 9개만 화면에 표시
        ax = plt.subplot(3, 3, i + 1)
        
        # 모델 예측 수행
        img_array = tf.expand_dims(images[i], 0) # 모델 입력을 위해 차원 추가
        predictions = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(predictions[0])]
        true_class = class_names[labels[i]]
        
        plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
        plt.title(f"True: {true_class}, Predicted: {predicted_class}")
        plt.axis("off")

plt.show()