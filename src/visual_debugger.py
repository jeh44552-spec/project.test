# src/visual_debugger.py

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 비교할 이미지 선택 ---
# hires_numbers 폴더에서 아무 숫자 폴더나 선택하고, 그 안의 이미지 파일 이름을 적으세요.
# 예: hires_numbers/1/100.jpg
TEST_IMAGE_PATH = 'hires_numbers/5/2lqgfZcq.png' 
# -------------------------

# --- 셰프 1: TensorFlow 최고급 셰프의 요리법 ---
img_tf = tf.keras.utils.load_img(
    TEST_IMAGE_PATH, target_size=(45, 45), color_mode="grayscale"
)
img_array_tf = tf.keras.utils.img_to_array(img_tf)
# ---------------------------------------------

# --- 셰프 2: 우리가 만든 OpenCV 셰프의 요리법 (현재 버전) ---
def preprocess_for_model_opencv(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 현재 predict_hires_hybrid.py 에 있는 로직 그대로
    padded = cv2.copyMakeBorder(img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(padded, (45, 45))
    return resized
# ----------------------------------------------------

img_cv2 = preprocess_for_model_opencv(TEST_IMAGE_PATH)


# --- 두 셰프의 결과물 비교 ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Chef 1: TensorFlow (The 'Correct' Food)")
plt.imshow(img_array_tf.reshape(45, 45), cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Chef 2: Our OpenCV (The 'Flawed' Food)")
plt.imshow(img_cv2, cmap='gray')

plt.show()

print("두 이미지가 다르게 보인다면, 그것이 바로 '무조건 8' 문제의 원인입니다!")
