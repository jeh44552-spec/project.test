# src/predict_transfer.py

import cv2
import numpy as np
import tensorflow as tf

# --- 최종 전이 학습 모델 로드 ---
try:
    model = tf.keras.models.load_model('../models/transfer_model.h5')
    class_names = np.load('../models/transfer_class_names.npy')
    class_names = [c if c != 'times' else 'x' for c in class_names]
    print("최종 전이 학습 모델 로드 성공!")
except OSError as e:
    print(f"모델 로딩 실패: {e}. 'train_transfer_learning.py'를 먼저 실행해 주세요.")
    exit()

# --- 전처리 함수 (3채널로 변경!) ---
def preprocess_for_model(img):
    resized = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA)
    # 흑백 이미지를 3채널로 변환
    img_rgb = cv.cvtColor(resized, cv.COLOR_GRAY2RGB)
    processed = img_rgb.reshape(1, 128, 128, 3).astype('float32') / 255.0
    return processed

cap = cv2.VideoCapture(0)
roi_x, roi_y, roi_w, roi_h = 100, 50, 600, 300 

while True:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_h]
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

        prediction = model.predict(input_data, verbose=0)[0]
        confidence = np.max(prediction)
        
        final_symbol = "?"
        if confidence > 0.6: # 60% 이상 확신하면 인정
            predicted_index = np.argmax(prediction)
            final_symbol = class_names[predicted_index]

        final_expression += final_symbol + " "
        cv2.rectangle(display_frame, (roi_x + x, roi_y + y), (roi_x + x + w, roi_y + y + h), (0, 0, 255), 2)
        cv2.putText(display_frame, final_symbol, (roi_x + x, roi_y + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(display_frame, f"Expression: {final_expression}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow('Transfer Learning Final Model', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()