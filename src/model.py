# src/model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def build_model():
    """
    손글씨 숫자 인식을 위한 간단한 신경망 모델을 생성하는 함수.
    
    반환 값:
        컴파일되지 않은 Keras 모델 객체
    """
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

if __name__ == '__main__':
    # 이 파일이 직접 실행될 때 테스트용으로 모델 구조 확인
    model = build_model()
    model.summary()