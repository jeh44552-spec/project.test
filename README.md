# 손글씨 숫자 인식 AI 프로젝트

이 프로젝트는 딥러닝의 "Hello, World!"로 불리는 MNIST 데이터셋을 사용하여 손으로 쓴 숫자를 인식하는 인공지능 모델을 개발합니다.

## 프로젝트 구조

```
mnist_project/
│
├── data/
├── notebooks/
│   └── 01_data_exploration_and_model_building.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── mnist_model.h5
│
├── README.md
└── requirements.txt
```

## 사용 기술

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## 설치 및 환경 설정

1.  이 저장소를 복제(clone)합니다.
    ```bash
    git clone <저장소_URL>
    cd mnist_project
    ```
2.  필요한 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

## 사용 방법

### 1. 모델 훈련

`src` 폴더로 이동하여 `train.py` 스크립트를 실행하면 모델 훈련이 시작되고, 훈련된 모델은 `models/mnist_model.h5` 파일로 저장됩니다.

```bash
cd src
python train.py
```

### 2. 예측 수행

훈련된 모델을 사용하여 새로운 이미지의 숫자를 예측하려면 `predict.py`를 실행합니다.

```bash
cd src
python predict.py
```