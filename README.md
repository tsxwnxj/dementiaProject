# 🧠 치매 예방 손가락 운동 인식 모델

보건복지부 권장 **7가지 손가락 운동**을 실시간으로 인식하고, 동작 정확도를 점수로 측정하는 PyTorch LSTM 기반 모델입니다.

---

## 📋 인식 동작 목록

| # | 동작명 | 폴더명 |
|---|--------|--------|
| 1 | 손가락 움직이기 | `finger_wave` |
| 2 | 손털기 | `hand_shake` |
| 3 | 손가락 접기 | `finger_fold` |
| 4 | 주먹 쥐고 펴기 | `fist_open` |
| 5 | 엇갈려 주먹 쥐고 펴기 | `cross_fist` |
| 6 | 손끝 박수 | `fingertip_clap` |
| 7 | 손사이 박수 | `side_clap` |

---

## 🏗️ 프로젝트 구조

```
dementiaGestureRecognition/
├── data/
│   └── sequences/          # 동작별 시퀀스 데이터 (.npy)
│       ├── finger_wave/
│       ├── hand_shake/
│       ├── finger_fold/
│       ├── fist_open/
│       ├── cross_fist/
│       ├── fingertip_clap/
│       └── side_clap/
├── model/
│   └── gesture_lstm.pt     # 학습된 LSTM 모델
├── src/
│   ├── collect_data.py     # 시퀀스 데이터 수집
│   ├── train_model.py      # LSTM 모델 학습
│   └── predict.py          # 실시간 예측 + 정확도 점수
├── requirements.txt
├── .gitignore
└── main.py
```

---

## ⚙️ 환경 설정

### 1. 가상환경 생성 (Python 3.11 권장)

```bash
conda create -n dementiaGesture python=3.11 -y
conda activate dementiaGesture
```

### 2. 라이브러리 설치

```bash
pip install -r requirements.txt
```

> **requirements.txt가 없을 경우** 아래 명령어로 직접 설치:
> ```bash
> pip install torch torchvision opencv-python "mediapipe==0.10.13" "numpy<2.0" scikit-learn matplotlib pandas
> ```

### 3. 설치 확인

```bash
python -c "import torch; import cv2; import mediapipe; print('all good')"
```

---

## 🚀 실행 순서

### Step 1. 데이터 수집

웹캠으로 각 동작당 50시퀀스(60프레임씩)를 수집합니다.

```bash
python src/collect_data.py
```

- 동작 이름이 화면에 표시되면 해당 동작을 수행하세요
- 동작당 약 5분 소요 (7개 동작 총 35분)
- 수집된 데이터는 `data/sequences/` 에 `.npy` 형식으로 저장됩니다

### Step 2. 모델 학습

```bash
python src/train_model.py
```

- 학습 완료 후 `model/gesture_lstm.pt` 가 생성됩니다
- Mac M1/M2/M3: MPS 자동 감지로 빠른 학습 가능

### Step 3. 실시간 예측

```bash
python src/predict.py
```

- 웹캠 화면에 인식된 동작명과 정확도 점수(%)가 표시됩니다
- 종료: `q` 키

---

## 🧬 모델 구조

```
입력: (60프레임, 126개 특징값)  ← 양손 21개 관절 × xyz × 2
        ↓
LSTM (hidden=128, layers=2, dropout=0.3)
        ↓
FC (128 → 64 → 7)
        ↓
출력: 7개 동작 분류 + Softmax 확률(정확도 점수)
```

---

## 팀원이 처음 세팅할 때

```bash
git clone <레포 주소>
cd dementiaGestureRecognition

conda create -n dementiaGesture python=3.11 -y
conda activate dementiaGesture
pip install -r requirements.txt
```

---

## 📦 주요 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| PyTorch | latest | LSTM 모델 학습/추론 |
| MediaPipe | 0.10.13 | 손 랜드마크 추출 |
| OpenCV | latest | 웹캠 영상 처리 |
| NumPy | <2.0 | 데이터 배열 처리 |
| scikit-learn | latest | 레이블 인코딩 |