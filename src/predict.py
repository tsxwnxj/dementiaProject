import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import pickle
from collections import deque, Counter

# ── 설정 ──────────────────────────────────────────────────
SEQUENCE_LEN         = 30
INPUT_SIZE           = 126
HIDDEN_SIZE          = 128
NUM_LAYERS           = 2
NUM_CLASSES          = 6
CONFIDENCE_THRESHOLD = 0.75   # 높일수록 확실할 때만 표시
HAND_DETECT_RATIO    = 0.7    # 버퍼 중 손 감지 비율 기준
SMOOTH_WINDOW        = 5      # 예측 스무딩 윈도우 크기

MODEL_DIR     = "/Users/jangjunseo/Desktop/dementiaProject/model"
LABEL_PATH    = f"{MODEL_DIR}/label_encoder.pkl"
ENSEMBLE_PATH = f"{MODEL_DIR}/ensemble_info.pkl"

GESTURE_EN = {
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
    def __init__(self):
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
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def load_ensemble(device):
    with open(ENSEMBLE_PATH, "rb") as f:
        info = pickle.load(f)
    models = []
    for path in info["model_paths"]:
        m = GestureLSTM().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
    print(f"[Loaded] 앙상블 모델 {len(models)}개")
    return models


def extract_landmarks(frame):
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
    bar_w = int(score / 100 * width)
    if score >= 80:
        color = (0, 255, 0)
    elif score >= 60:
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + 20), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + 20), (200, 200, 200), 2)
    cv2.putText(frame, f"{score}%", (x + width + 10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return color


def predict():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with open(LABEL_PATH, "rb") as f:
        le = pickle.load(f)

    models = load_ensemble(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    buffer        = deque(maxlen=SEQUENCE_LEN)   # 랜드마크 버퍼
    pred_history  = deque(maxlen=SMOOTH_WINDOW)  # 예측 스무딩용

    gesture_name = ""
    score        = 0

    print("=" * 50)
    print("  Dementia Prevention - Gesture Recognition")
    print(f"  Ensemble ({len(models)} models) | Confidence >= {CONFIDENCE_THRESHOLD:.0%}")
    print("  Press 'q' to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        landmarks = extract_landmarks(frame)
        buffer.append(landmarks)

        if len(buffer) == SEQUENCE_LEN:

            # ── 1. 손 감지 비율 확인 ──────────────────────
            hand_detected = sum(1 for f in buffer if np.any(f != 0))
            hand_ratio    = hand_detected / SEQUENCE_LEN

            if hand_ratio < HAND_DETECT_RATIO:
                # 손이 충분히 감지 안 됨 → 초기화
                gesture_name = "..."
                score        = 0
                pred_history.clear()

            else:
                # ── 2. 앙상블 예측 ────────────────────────
                seq = torch.tensor(
                    np.array(buffer), dtype=torch.float32
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs = torch.zeros(1, NUM_CLASSES).to(device)
                    for m in models:
                        probs += F.softmax(m(seq), dim=1)
                    probs /= len(models)

                    conf, idx = probs.max(dim=1)
                    conf_val  = conf.item()

                if conf_val >= CONFIDENCE_THRESHOLD:
                    pred_history.append(idx.item())

                    # ── 3. 스무딩: 최근 N개 중 최빈값 ──────
                    most_common_idx = Counter(pred_history).most_common(1)[0][0]
                    gesture_key     = le.inverse_transform([most_common_idx])[0]
                    gesture_name    = GESTURE_EN.get(gesture_key, gesture_key)
                    score           = int(conf_val * 100)
                else:
                    gesture_name = "..."
                    score        = 0

        # ── UI 렌더링 ──────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 175), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, gesture_name, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        if score > 0:
            color = draw_score_bar(frame, score)
            if score >= 80:
                msg = "Great!"
            elif score >= 60:
                msg = "Keep going!"
            else:
                msg = "Try again!"
            cv2.putText(frame, msg, (20, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 손 감지 비율 표시 (디버깅용)
        if len(buffer) == SEQUENCE_LEN:
            ratio_text = f"Hand: {hand_ratio:.0%}"
            ratio_color = (0, 255, 0) if hand_ratio >= HAND_DETECT_RATIO else (0, 0, 255)
            cv2.putText(frame, ratio_text, (20, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ratio_color, 1)

        cv2.imshow("Dementia Prevention - Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict()