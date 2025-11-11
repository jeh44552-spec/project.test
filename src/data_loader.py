# src/data_loader.py

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    """
    MNIST 데이터셋을 로드하고 전처리하는 함수.
    
    반환 값:
        (x_train, y_train_cat), (x_test, y_test_cat)
    """
    # 1. 데이터 로드
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. 이미지 데이터 정규화
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 3. 라벨 데이터 원-핫 인코딩
    y_train_cat = to_categorical(y_train, num_classes=10)
    y_test_cat = to_categorical(y_test, num_classes=10)
    
    return (x_train, y_train_cat), (x_test, y_test_cat)

if __name__ == '__main__':
    # 이 파일이 직접 실행될 때 테스트용으로 데이터 로딩 확인
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print("데이터 로딩 및 전처리 완료!")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")