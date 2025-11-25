# 손글씨 수식 인식 AI 계산기

이 프로젝트는 실시간 카메라 입력을 통해 손으로 쓴 숫자와 간단한 수식을 인식하는 AI 모델을 개발합니다.

## 설치 및 사용 방법

### 1. 저장소 복제 (Clone Repository)
```bash
git clone https://github.com/당신의-깃허브-ID/project.test-main.git
cd project.test-main
```

### 2. 가상환경 설정 및 라이브러리 설치
```bash
# 가상환경 생성
python -m venv venv
.\venv\Scripts\activate
<<<<<<< HEAD
# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 필요한 라이브러리 설치
pip install -r requirements.txt
```

### 3. 데이터셋 준비
이 프로젝트는 [Kaggle의 Handwritten Math Symbols 데이터셋](https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols)을 기반으로 합니다.

1.  위 링크에서 데이터셋을 다운로드합니다.
2.  프로젝트 폴더 안에 `final_dataset` 이라는 이름의 새 폴더를 만듭니다.
3.  다운로드한 데이터셋에서 `0`~`9`, `+`, `-`, `times` 폴더를 `final_dataset` 폴더 안으로 복사합니다.

### 4. AI 모델 훈련 (중요!)
이 저장소에는 훈련된 모델 파일이 포함되어 있지 않습니다. 아래 명령어를 실행하여 직접 모델을 훈련시켜야 합니다. 이 과정은 컴퓨터 사양에 따라 수십 분 이상 소요될 수 있습니다.

```bash
# src 폴더로 이동
cd src

# 최종 모델 훈련 스크립트 실행
python train_ultimate.py
```

### 5. 애플리케이션 실행
모델 훈련이 완료되면, 아래 명령어를 실행하여 실시간 인식 프로그램을 시작할 수 있습니다.

```bash
# (src 폴더 안에서)
python predict_ultimate.py
```
=======
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
>>>>>>> 36f5d0e643e17d8e9e58dd2f697228cd026d8ea6
wq
