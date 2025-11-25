# src/predict_hires_hybrid_128.py (ASYMMETRIC THRESHOLD VERSION)

import cv2
import numpy as np
import tensorflow as tf

# --- 모델 로드 (변경 없음) ---
try:
    number_model = tf.keras.models.load_model('../models/hires_number_model_128.h5')
    symbol_model = tf.keras.models.load_model('../models/hires_symbol_model_128.h5')
    number_class_names = np.load('../models/hires_number_class_names_128.npy')
    symbol_class_names = np.load('../models/hires_symbol_class_names_128.npy')
    symbol_class_names = [c if c != 'times' else 'x' for c in symbol_class_names] # 'x'로 표시
    print("초고해상도 전문가 모델 모두 로드 성공!")
except OSError as e:
    print(f"모델 로딩 실패: {e}.")
    exit()

# --- 전처리 함수 (변경 없음) ---
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
        num_pred = number_model.predict(input_data, verbose=0)
        sym_pred = symbol_model.predict(input_data, verbose=0)
        
        # --- *** 비대칭 자격 심사 로직 *** ---
        num_pred_1d = num_pred[0]
        sym_pred_1d = sym_pred[0]
        num_top1_conf = np.max(num_pred_1d)
        sym_top1_conf = np.max(sym_pred_1d)

        # 1. 각 전문가에게 서로 다른, 맞춤형 자격 기준을 부여합니다.
        # 숫자 전문가는 조금 더 관대하게 (70점 이상이면 발언권)
        NUMBER_QUALIFICATION_THRESHOLD = 0.70 
        # 기호 전문가는 조금 더 엄격하게 (95점 이상일 때만 발언권)
        SYMBOL_QUALIFICATION_THRESHOLD = 0.95 

        is_number_qualified = num_top1_conf > NUMBER_QUALIFICATION_THRESHOLD
        is_symbol_qualified = sym_top1_conf > SYMBOL_QUALIFICATION_THRESHOLD
        
        final_symbol = "?" 

        if is_number_qualified and not is_symbol_qualified:
            final_symbol = number_class_names[np.argmax(num_pred_1d)]
        elif is_symbol_qualified and not is_number_qualified:
            final_symbol = symbol_class_names[np.argmax(sym_pred_1d)]
        elif is_number_qualified and is_symbol_qualified:
            if num_top1_conf > sym_top1_conf:
                final_symbol = number_class_names[np.argmax(num_pred_1d)]
            else:
                final_symbol = symbol_class_names[np.argmax(sym_pred_1d)]
        else:
            final_symbol = "?"

        # 2. 디버깅 출력
        print("---")
        print(f"Number Conf: {num_top1_conf:.4f} | Qualified: {is_number_qualified} (>{NUMBER_QUALIFICATION_THRESHOLD})")
        print(f"Symbol Conf: {sym_top1_conf:.4f} | Qualified: {is_symbol_qualified} (>{SYMBOL_QUALIFICATION_THRESHOLD})")
        print(f"--> Predicted Symbol: {final_symbol}")
        # ----------------------------------------

        final_expression += final_symbol + " "
        cv2.rectangle(display_frame, (roi_x + x, roi_y + y), (roi_x + x + w, roi_y + y + h), (0, 0, 255), 2)
        cv2.putText(display_frame, final_symbol, (roi_x + x, roi_y + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(display_frame, f"Expression: {final_expression}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow('Final Tuned Hybrid Model', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()