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

# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
.\venv\Scripts\activate

pip install -r requirements.txt

# src 폴더로 이동
cd src

# 훈련 스크립트 실행
python train.py

# (이미 src 폴더 안에 있으므로 바로 실행)
# 이미지 파일로 예측
python predict.py

# 카메라로 예측 (카메라가 연결된 로컬 컴퓨터일 경우)
python predict_multi_digit.py


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

