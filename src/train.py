# src/train.py

import os
from data_loader import load_and_preprocess_data
from model import build_model

def main():
    # 1. 데이터 로딩
    print("데이터를 로드합니다...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # 2. 모델 구축
    print("모델을 구축합니다...")
    model = build_model()
    
    # 3. 모델 컴파일
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 4. 모델 훈련
    print("모델 훈련을 시작합니다...")
    model.fit(x_train, y_train, 
              epochs=5, 
              batch_size=32, 
              validation_split=0.2)
    
    # 5. 모델 평가
    print("\n모델 평가를 시작합니다...")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"테스트 데이터 정확도: {accuracy*100:.2f}%")
    
    # 6. 모델 저장
    # 'models' 디렉터리가 없으면 생성
    if not os.path.exists('../models'):
        os.makedirs('../models')
        
    model.save('../models/mnist_model.h5')
    print("모델이 'models/mnist_model.h5' 파일로 저장되었습니다.")


if __name__ == '__main__':
    main()