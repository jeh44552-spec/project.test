# src/predict_multi_digit.py

import cv2
import numpy as np
import tensorflow as tf

# 1. 훈련된 모델 로드
MODEL_PATH = '../models/mnist_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# 2. 웹캠 설정
#밑에 괄호에 0이면 기본 노트북 웹캠/ 1이면 별도 장착한 카메라
cap = cv2.VideoCapture(0)

# 화면에 숫자를 쓸 영역(Region of Interest, ROI) 설정
roi_x, roi_y, roi_w, roi_h = 250, 100, 400, 200

def preprocess_digit(digit_img):
    """
    잘라낸 숫자 이미지를 MNIST 모델 입력 형식에 맞게 전처리하는 함수.
    """
    # MNIST는 28x28 크기, 흑백 이미지
    resized_digit = cv2.resize(digit_img, (28, 28))

    # 모델이 학습한 데이터처럼 보이도록 전처리 (정규화, 차원 확장)
    processed_digit = resized_digit.astype('float32') / 255.0
    processed_digit = np.expand_dims(processed_digit, axis=0) # (1, 28, 28)
    
    return processed_digit

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 좌우 반전 (거울 모드)
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # 3. ROI 영역을 화면에 표시
    cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    
    # 4. ROI 영역 이미지 전처리 (Contour 찾기 준비)
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 배경은 검은색, 글씨는 흰색으로 반전 및 이진화
    _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

    # 5. 이미지에서 모든 숫자 덩어리(Contour) 찾기
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for c in contours:
        # 너무 작은 노이즈는 무시
        if cv2.contourArea(c) > 50:
            (x, y, w, h) = cv2.boundingRect(c)
            bounding_boxes.append((x, y, w, h))

    # 6. x 좌표 기준으로 왼쪽에서 오른쪽으로 정렬
    bounding_boxes.sort(key=lambda b: b[0])
    
    final_number = ""
    
    # 7. 정렬된 각 숫자에 대해 예측 수행
    for (x, y, w, h) in bounding_boxes:
        # 원본 ROI(이진화 이미지)에서 숫자 부분만 잘라내기
        digit_img = binary[y:y+h, x:x+w]
        
        # 모델 입력에 맞게 전처리
        processed_digit = preprocess_digit(digit_img)
        
        # 예측
        prediction = model.predict(processed_digit)
        predicted_digit = np.argmax(prediction)
        
        final_number += str(predicted_digit)
        
        # 화면에 인식된 숫자와 경계 상자 표시
        cv2.rectangle(display_frame, (roi_x + x, roi_y + y), (roi_x + x + w, roi_y + y + h), (0, 0, 255), 2)
        cv2.putText(display_frame, str(predicted_digit), (roi_x + x, roi_y + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 8. 최종 인식된 숫자를 화면에 표시
    cv2.putText(display_frame, f"Detected Number: {final_number}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    cv2.imshow('Multi-digit Recognition', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
