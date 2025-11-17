# MNIST 손글씨 숫자 인식 프로젝트

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

딥러닝의 "Hello, World!"로 불리는 MNIST 데이터셋을 활용한 손글씨 숫자 인식 AI 모델 프로젝트입니다.

## 📋 목차

- [프로젝트 소개](#프로젝트-소개)
- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [기술 스택](#기술-스택)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [모델 정보](#모델-정보)
- [예제](#예제)
- [기여 방법](#기여-방법)
- [라이센스](#라이센스)

## 🎯 프로젝트 소개

이 프로젝트는 MNIST(Modified National Institute of Standards and Technology) 데이터셋을 사용하여 0-9까지의 손글씨 숫자를 인식하는 딥러닝 모델을 구현합니다. 

### MNIST 데이터셋이란?

- 60,000개의 훈련 이미지와 10,000개의 테스트 이미지로 구성
- 28x28 픽셀의 흑백 이미지
- 손으로 쓴 숫자 0-9를 포함
- 머신러닝/딥러닝 입문자들을 위한 표준 벤치마크 데이터셋

## ✨ 주요 기능

- **모델 훈련**: TensorFlow/Keras를 사용한 CNN 모델 구축 및 훈련
- **단일 숫자 예측**: 개별 이미지에서 숫자 인식
- **다중 숫자 예측**: 여러 숫자가 포함된 이미지 처리
- **데이터 탐색**: Jupyter Notebook을 통한 데이터 분석 및 시각화
- **모델 저장/로드**: 훈련된 모델의 영구 저장 및 재사용

## 📁 프로젝트 구조

```
mnist_project/
│
├── data/                    # 데이터셋 저장 디렉토리
│   └── (MNIST 데이터 자동 다운로드)
│
├── notebooks/               # Jupyter 노트북
│   └── 01_data_exploration_and_model_building.ipynb
│
├── src/                     # 소스 코드
│   ├── __init__.py         # 패키지 초기화 파일
│   ├── data_loader.py      # 데이터 로딩 및 전처리
│   ├── model.py            # 모델 아키텍처 정의
│   ├── train.py            # 모델 훈련 스크립트
│   ├── predict.py          # 단일 숫자 예측
│   └── predict_multi_digit.py  # 다중 숫자 예측
│
├── models/                  # 훈련된 모델 저장
│   └── mnist_model.h5      # 저장된 모델 파일
│
├── README.md               # 프로젝트 문서
└── requirements.txt        # 필요한 패키지 목록
```

## 🛠 기술 스택

### 핵심 라이브러리

- **Python 3.8+**: 프로그래밍 언어
- **TensorFlow 2.x / Keras**: 딥러닝 프레임워크
- **NumPy**: 수치 계산 및 배열 처리
- **Matplotlib**: 데이터 시각화
- **Pandas**: 데이터 분석 및 조작 (선택사항)

### 개발 도구

- Jupyter Notebook: 대화형 개발 환경
- Git: 버전 관리

## 🚀 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/jeh44552-spec/project.test.git
cd project.test
```

### 2. 가상환경 생성 및 활성화

#### Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

## 💻 사용 방법

### 모델 훈련

`src` 디렉토리로 이동하여 `train.py`를 실행합니다. 훈련이 완료되면 모델이 `models/mnist_model.h5`에 저장됩니다.

```bash
cd src
python train.py
```

**예상 출력:**
- 훈련 진행 상황 (에포크별 정확도 및 손실)
- 최종 테스트 정확도
- 모델 저장 확인 메시지

### 단일 숫자 예측

훈련된 모델을 사용하여 새로운 이미지의 숫자를 예측합니다.

```bash
cd src
python predict.py
```

### 다중 숫자 예측

여러 숫자가 포함된 이미지를 처리합니다.

```bash
cd src
python predict_multi_digit.py
```

### Jupyter Notebook으로 탐색

데이터 탐색 및 모델 구축 과정을 대화형으로 확인할 수 있습니다.

```bash
jupyter notebook notebooks/01_data_exploration_and_model_building.ipynb
```

## 🧠 모델 정보

### 모델 아키텍처

일반적인 CNN(Convolutional Neural Network) 구조:

```
입력층 (28x28x1)
    ↓
합성곱층 1 + ReLU + MaxPooling
    ↓
합성곱층 2 + ReLU + MaxPooling
    ↓
Flatten
    ↓
완전연결층 (Dense) + Dropout
    ↓
출력층 (10개 클래스, Softmax)
```

### 성능 지표

- **예상 정확도**: 약 98-99% (테스트 데이터셋 기준)
- **훈련 시간**: CPU 기준 약 5-10분
- **모델 크기**: 약 1-2MB

## 📊 예제

### 예측 결과 예시

```python
# 예측된 숫자: 7
# 신뢰도: 99.87%
```

## 🤝 기여 방법

프로젝트에 기여하고 싶으시다면:

1. 이 저장소를 포크합니다
2. 새로운 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 📝 개선 아이디어

- [ ] 모델 성능 개선 (더 깊은 네트워크, 하이퍼파라미터 튜닝)
- [ ] 웹 인터페이스 추가 (Flask/Streamlit)
- [ ] 실시간 손글씨 입력 기능
- [ ] 모델 경량화 (모바일 배포용)
- [ ] 데이터 증강(Data Augmentation) 적용

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 👨‍💻 개발자

**jeh44552-spec**
- GitHub: [@jeh44552-spec](https://github.com/jeh44552-spec)

## 📚 참고 자료

- [MNIST 데이터셋 공식 페이지](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow 공식 문서](https://www.tensorflow.org/)
- [Keras 공식 문서](https://keras.io/)

---

⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!
