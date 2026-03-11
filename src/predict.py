import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import pickle
from collections import deque

# ── 설정 ──────────────────────────────────────────────────
SEQUENCE_LEN = 30
INPUT_SIZE   = 126
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
MODEL_PATH   = "model/gesture_lstm.pt"
LABEL_PATH   = "model/label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.5  # 이 이하면 인식 안된 것으로 처리

GESTURE_KO = {
    "finger_wave":    "1. Finger Wave",
    "hand_shake":     "2. Hand Shake",
    "finger_fold":    "3. Finger Fold",
    "fist_open":      "4. Fist Open",
    "cross_fist":     "5. Cross Fist",
    "fingertip_clap": "6. Fingertip Clap",
}

# ── MediaPipe 초기화 ───────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ── 모델 ──────────────────────────────────────────────────
class GestureLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.3,
        )
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def extract_landmarks(frame):
    """프레임에서 양손 랜드마크 추출 → (126,) 배열"""
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    data   = np.zeros(126, dtype=np.float32)

    if result.multi_hand_landmarks:
        for i, hand_lm in enumerate(result.multi_hand_landmarks[:2]):
            offset = i * 63
            for j, lm in enumerate(hand_lm.landmark):
                data[offset + j * 3    ] = lm.x
                data[offset + j * 3 + 1] = lm.y
                data[offset + j * 3 + 2] = lm.z
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

    return data


def draw_score_bar(frame, score, x=20, y=130, width=300):
    """정확도 점수 바 시각화"""
    bar_w = int(score / 100 * width)
    if score >= 80:
        color = (0, 255, 0)      # 초록: 잘하고 있어요
    elif score >= 60:
        color = (0, 165, 255)    # 주황: 보통
    else:
        color = (0, 0, 255)      # 빨강: 다시 해보세요

    cv2.rectangle(frame, (x, y), (x + bar_w, y + 20), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + 20), (200, 200, 200), 2)
    cv2.putText(frame, f"{score}%", (x + width + 10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return color


def predict():
    # 레이블 인코더 로드
    with open(LABEL_PATH, "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)

    # 모델 로드
    model = GestureLSTM(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    buffer = deque(maxlen=SEQUENCE_LEN)  # 슬라이딩 윈도우

    print("=" * 50)
    print("  Dementia Prevention - Gesture Recognition")
    print("  Press 'q' to quit")
    print("=" * 50)

    gesture_name = ""
    score        = 0
    color        = (200, 200, 200)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # 랜드마크 추출 & 버퍼에 추가
        landmarks = extract_landmarks(frame)
        buffer.append(landmarks)

        # 버퍼가 가득 찼을 때만 예측
        if len(buffer) == SEQUENCE_LEN:
            seq = torch.tensor(
                np.array(buffer), dtype=torch.float32
            ).unsqueeze(0)  # (1, 60, 126)

            with torch.no_grad():
                probs = F.softmax(model(seq), dim=1)
                conf, idx = probs.max(dim=1)
                conf_val = conf.item()

            if conf_val >= CONFIDENCE_THRESHOLD:
                gesture_key  = le.inverse_transform([idx.item()])[0]
                gesture_name = GESTURE_KO.get(gesture_key, gesture_key)
                score        = int(conf_val * 100)
            else:
                gesture_name = "..."
                score        = 0

        # ── UI 렌더링 ──
        # 반투명 배경 박스
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 동작명
        cv2.putText(frame, gesture_name, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        # 정확도 바
        if score > 0:
            color = draw_score_bar(frame, score)
            # 피드백 메시지
            if score >= 80:
                msg = "Great!"
            elif score >= 60:
                msg = "Keep going!"
            else:
                msg = "Try again!"
            cv2.putText(frame, msg, (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Dementia Prevention - Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict()