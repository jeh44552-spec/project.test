# src/check_class_order.py

import tensorflow as tf

DATASET_PATH = '../hires_numbers'

# 훈련 때와 똑같은 방식으로 데이터셋을 불러와서
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(45, 45),
    color_mode='grayscale'
)

# 숨겨진 진짜 클래스 순서를 출력한다!
print("훈련 시 사용된 실제 클래스 순서:")
print(val_ds.class_names)