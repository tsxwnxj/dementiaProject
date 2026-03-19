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
HIDDEN_SIZE          = 64
NUM_LAYERS           = 2
NUM_CLASSES          = 6
CONFIDENCE_THRESHOLD = 0.75
HAND_DETECT_RATIO    = 0.7
SMOOTH_WINDOW        = 5

MODEL_DIR  = "/Users/jangjunseo/Desktop/dementiaProject/model"
MODEL_PATH = f"{MODEL_DIR}/gesture_cnn_lstm.pt"
LABEL_PATH = f"{MODEL_DIR}/label_encoder.pkl"

GESTURE_KO = {
    "finger_wave":    "finger_wave",
    "hand_shake":     "hand_shake",
    "finger_fold":    "finger_fold",
    "fist_open":      "fist_open",
    "cross_fist":     "cross_fist",
    "fingertip_clap": "fingertip_clap",
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


# ── CNN-LSTM 모델 ──────────────────────────────────────────
class GestureCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(INPUT_SIZE, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(
            input_size=128,
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
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


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


def predict():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with open(LABEL_PATH, "rb") as f:
        le = pickle.load(f)

    model = GestureCNNLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[Loaded] CNN-LSTM 모델 → {MODEL_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    buffer       = deque(maxlen=SEQUENCE_LEN)
    pred_history = deque(maxlen=SMOOTH_WINDOW)
    gesture_name = ""
    hand_ratio   = 0.0

    print("=" * 50)
    print("  치매 예방 손가락 운동 인식 [CNN-LSTM]")
    print("  Press 'q' to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        landmarks = extract_landmarks(frame)
        buffer.append(landmarks)

        if len(buffer) == SEQUENCE_LEN:
            hand_detected = sum(1 for f in buffer if np.any(f != 0))
            hand_ratio    = hand_detected / SEQUENCE_LEN

            if hand_ratio < HAND_DETECT_RATIO:
                gesture_name = ""
                pred_history.clear()
            else:
                seq = torch.tensor(
                    np.array(buffer), dtype=torch.float32
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs     = F.softmax(model(seq), dim=1)
                    conf, idx = probs.max(dim=1)
                    conf_val  = conf.item()

                if conf_val >= CONFIDENCE_THRESHOLD:
                    pred_history.append(idx.item())
                    most_common = Counter(pred_history).most_common(1)[0][0]
                    gesture_key = le.inverse_transform([most_common])[0]
                    gesture_name = GESTURE_KO.get(gesture_key, gesture_key)
                else:
                    gesture_name = ""

        # ── UI 렌더링 ──────────────────────────────────────
        if gesture_name:
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness  = 2
            (text_w, text_h), _ = cv2.getTextSize(
                gesture_name, font, font_scale, thickness)

            box_x1 = 20
            box_y1 = h - 80
            box_x2 = box_x1 + text_w + 30
            box_y2 = h - 20

            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, gesture_name,
                        (box_x1 + 15, box_y2 - 15),
                        font, font_scale, (0, 255, 0), thickness)

        ratio_color = (0, 255, 0) if hand_ratio >= HAND_DETECT_RATIO else (0, 0, 255)
        cv2.putText(frame, f"Hand: {hand_ratio:.0%}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ratio_color, 1)

        cv2.imshow("치매 예방 손가락 운동 인식", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict()