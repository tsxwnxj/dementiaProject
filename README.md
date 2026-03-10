# dementiaProject

# 설치 명령어 (dementiaGesture 가상환경 활성화 후)
pip install torch torchvision opencv-python "mediapipe==0.10.13" \
            "numpy<2.0" scikit-learn matplotlib pandas

# 폴더 구조
dementiaGestureRecognition/
├── data/
│   └── sequences/          # 동작별 시퀀스 데이터 (.npy)
│       ├── finger_wave/     # 1. 손가락 움직이기
│       ├── hand_shake/      # 2. 손털기
│       ├── finger_fold/     # 3. 손가락 접기
│       ├── fist_open/       # 4. 주먹 쥐고 펴기
│       ├── cross_fist/      # 5. 엇갈려 주먹 쥐고 펴기
│       ├── fingertip_clap/  # 6. 손끝 박수
│       └── side_clap/       # 7. 손사이 박수
├── model/
│   └── gesture_lstm.pt     # 학습된 LSTM 모델
├── src/
│   ├── collect_data.py     # 시퀀스 데이터 수집
│   ├── train_model.py      # LSTM 모델 학습
│   └── predict.py          # 실시간 예측 + 정확도 점수
└── main.py
