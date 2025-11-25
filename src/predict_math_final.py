# src/predict_math_final.py

import cv2
import numpy as np
import tensorflow as tf

# 1. 훈련된 수학 기호 모델과 클래스 이름 로드
try:
    model = tf.keras.models.load_model('../models/math_symbol_model.h5')
    class_names = np.load('../models/math_class_names.npy')
    print("수학 기호 모델 및 클래스 이름 로드 성공!")
    # 'times'를 'x'로 바꿔주면 더 직관적임
    class_names = [c if c != 'times' else '*' for c in class_names]
except OSError:
    print("모델 또는 클래스 이름 파일을 찾을 수 없습니다.")
    print("먼저 'train_math_symbols.py'를 실행하여 모델을 훈련시켜 주세요.")
    exit()

# 웹캠 설정 및 ROI
cap = cv2.VideoCapture(1)
roi_x, roi_y, roi_w, roi_h = 150, 100, 500, 200

while True:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        # 필터링 조건: 너무 작거나, 너무 넓적한 노이즈는 무시
        if w*h > 300 and h > 10: 
            bounding_boxes.append((x, y, w, h))

    bounding_boxes.sort(key=lambda b: b[0])
    
    final_expression = ""

    for (x, y, w, h) in bounding_boxes:
        digit_img = binary[y:y+h, x:x+w]
        
        # 모델 입력 크기(45x45)에 맞게 리사이즈 및 전처리
        padded = cv2.copyMakeBorder(digit_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        resized = cv2.resize(padded, (45, 45))
        processed = resized.reshape(1, 45, 45, 1).astype('float32') / 255.0
        
        # 예측
        prediction = model.predict(processed, verbose=0)
        predicted_index = np.argmax(prediction)
        
        # **핵심 업그레이드**: 예측된 인덱스를 실제 기호로 변환!
        predicted_symbol = class_names[predicted_index]
        final_expression += predicted_symbol + " "
        
        cv2.rectangle(display_frame, (roi_x + x, roi_y + y), (roi_x + x + w, roi_y + y + h), (0, 0, 255), 2)
        cv2.putText(display_frame, predicted_symbol, (roi_x + x, roi_y + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(display_frame, f"Expression: {final_expression}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow('Math Symbol Recognition', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()