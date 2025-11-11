# src/predict.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def predict_digit(model_path, image):
    """
    저장된 모델을 불러와 주어진 이미지에 대해 예측을 수행하는 함수.
    
    매개변수:
        model_path (str): 저장된 모델 파일 경로 (.h5)
        image (numpy.ndarray): 예측할 28x28 이미지
        
    반환 값:
        예측된 숫자 (int)
    """
    # 1. 모델 로드
    model = tf.keras.models.load_model(model_path)
    
    # 2. 이미지 차원 확장 (모델 입력 형식에 맞게)
    # (28, 28) -> (1, 28, 28)
    image_expanded = np.expand_dims(image, axis=0)
    
    # 3. 예측 수행
    prediction = model.predict(image_expanded)
    
    # 4. 가장 높은 확률을 가진 클래스(숫자) 반환
    predicted_digit = np.argmax(prediction)
    
    return predicted_digit

if __name__ == '__main__':
    # 테스트용 데이터 로드
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0  # 정규화

    # 예측할 이미지 선택 (예: 테스트 데이터의 첫 번째 이미지)
    test_image = x_test[0]
    true_label = y_test[0]
    
    # 모델 경로
    MODEL_PATH = '../models/mnist_model.h5'
    
    # 예측 수행
    predicted_digit = predict_digit(MODEL_PATH, test_image)
    
    # 결과 출력 및 시각화
    print(f"예측된 숫자: {predicted_digit}")
    print(f"실제 숫자: {true_label}")

    plt.imshow(test_image, cmap='gray')
    plt.title(f"Predicted: {predicted_digit}, True: {true_label}")
    plt.show()