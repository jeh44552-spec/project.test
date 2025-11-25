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

# 가상환경 활성화 (Windows)
.\venv\Scripts\activate
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