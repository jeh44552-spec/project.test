# src/predict_final.py

import cv2
import numpy as np
import tensorflow as tf

# 1. 훈련된 '향상된' 모델 로드
MODEL_PATH = '../models/mnist_cnn_augmented_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("향상된 모델 로드 성공!")
except OSError:
    print(f"'{MODEL_PATH}'에서 모델을 찾을 수 없습니다.")
    print("먼저 'train_upgraded.py'를 실행하여 모델을 훈련시켜 주세요.")
    exit()

# 웹캠 설정
cap = cv2.VideoCapture(0)
roi_x, roi_y, roi_w, roi_h = 200, 100, 400, 200

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # [해결 A] 좌우 반전(거울 모드)을 원하지 않으면 아래 줄을 주석 처리(#) 하세요.
    # frame = cv2.flip(frame, 1)
    
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    
    # ROI 영역 전처리
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # [개선 C] 조명 변화에 강한 적응형 스레시홀드 사용
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 모든 숫자 덩어리(Contour) 찾기
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        # [개선 A] 더 똑똑한 필터링: 크기와 종횡비(세로가 너비의 1.2배 이상) 동시 체크
        if w*h > 800 and h > w * 1.2:
            bounding_boxes.append((x, y, w, h))

    bounding_boxes.sort(key=lambda b: b[0])
    
    final_number = ""
    ai_inputs = [] # AI가 보는 이미지를 저장할 리스트

    for (x, y, w, h) in bounding_boxes:
        digit_img = binary[y:y+h, x:x+w]
        
        # 여백을 추가하여 MNIST와 유사한 형태로 만듦 (중요!)
        padded_digit = cv2.copyMakeBorder(digit_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        
        # 모델 입력 형식에 맞게 리사이즈 및 전처리
        resized_digit = cv2.resize(padded_digit, (28, 28))
        processed_digit = resized_digit.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # 예측
        prediction = model.predict(processed_digit, verbose=0)
        final_number += str(np.argmax(prediction))
        
        # [개선 B] AI가 보는 이미지를 나중에 표시하기 위해 저장
        ai_inputs.append(resized_digit)
        
        cv2.rectangle(display_frame, (roi_x + x, roi_y + y), (roi_x + x + w, roi_y + y + h), (0, 0, 255), 2)

    cv2.putText(display_frame, f"Detected: {final_number}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    # [개선 B] AI가 보는 화면을 새 창으로 보여주기
    if ai_inputs:
        # 모든 입력 이미지를 가로로 연결하여 하나로 만듦
        combined_ai_view = cv2.hconcat(ai_inputs)
        cv2.imshow("What AI Sees (Input)", combined_ai_view)
    else:
        # 숫자가 없을 때 창을 닫거나 빈 화면 표시
        # cv2.destroyWindow("What AI Sees (Input)") # 창을 닫는 옵션
        pass

    cv2.imshow('Final Recognition App', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()