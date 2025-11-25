# src/predict_final.py

import cv2
import numpy as np
import tensorflow as tf

# --- 최종 마스터 모델 하나만 로드 ---
try:
    model = tf.keras.models.load_model('../models/final_model.h5')
    class_names = np.load('../models/final_class_names.npy')
    class_names = [c if c != 'times' else 'x' for c in class_names]
    print("최종 마스터 전문가 모델 로드 성공!")
except OSError as e:
    print(f"모델 로딩 실패: {e}. 'train_final_model.py'를 먼저 실행해 주세요.")
    exit()

# --- 전처리 함수 (128x128) ---
def preprocess_for_model(img):
    resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    processed = resized.reshape(1, 128, 128, 1).astype('float32') / 255.0
    return processed

cap = cv2.VideoCapture(0)
roi_x, roi_y, roi_w, roi_h = 100, 50, 600, 300 

while True:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 5)

    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 200]
    bounding_boxes.sort(key=lambda b: b[0])
    
    final_expression = ""

    for (x, y, w, h) in bounding_boxes:
        digit_img = binary[y:y+h, x:x+w]
        input_data = preprocess_for_model(digit_img)

        # --- 훨씬 더 간단해진 예측 로직 ---
        # 1. 마스터 전문가에게 한 번만 물어봅니다.
        prediction = model.predict(input_data, verbose=0)[0]
        confidence = np.max(prediction)
        
        final_symbol = "?"
        # 2. 최소한의 자신감(50%)만 넘으면, 결과를 그대로 믿습니다.
        if confidence > 0.5:
            predicted_index = np.argmax(prediction)
            final_symbol = class_names[predicted_index]
        # -----------------------------------

        final_expression += final_symbol + " "
        cv2.rectangle(display_frame, (roi_x + x, roi_y + y), (roi_x + x + w, roi_y + y + h), (0, 0, 255), 2)
        cv2.putText(display_frame, final_symbol, (roi_x + x, roi_y + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(display_frame, f"Expression: {final_expression}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow('The Final Master Model', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()