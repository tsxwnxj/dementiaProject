import os
import cv2
import numpy as np
import mediapipe as mp

# ── 설정 ──────────────────────────────────────────────────
GESTURES = [
    "finger_wave",     # 1. 손가락 움직이기
    "hand_shake",      # 2. 손털기
    "finger_fold",     # 3. 손가락 접기
    "fist_open",       # 4. 주먹 쥐고 펴기
    "cross_fist",      # 5. 엇갈려 주먹 쥐고 펴기
    "fingertip_clap",  # 6. 손끝 박수
]
SEQUENCE_LEN  = 30   # 동작 1회 = 60프레임 (~2초)
NUM_SEQUENCES = 50  # 동작당 50시퀀스
DATA_PATH     = "data/sequences"

# ── MediaPipe 초기화 ───────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ── 폴더 생성 ──────────────────────────────────────────────
os.makedirs(DATA_PATH, exist_ok=True)
for g in GESTURES:
    os.makedirs(os.path.join(DATA_PATH, g), exist_ok=True)


def extract_landmarks(frame):
    """프레임에서 양손 랜드마크 추출 → (126,) 배열 반환"""
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    data   = np.zeros(126, dtype=np.float32)  # 손 없으면 zeros

    if result.multi_hand_landmarks:
        for i, hand_lm in enumerate(result.multi_hand_landmarks[:2]):
            offset = i * 63
            for j, lm in enumerate(hand_lm.landmark):
                data[offset + j * 3    ] = lm.x
                data[offset + j * 3 + 1] = lm.y
                data[offset + j * 3 + 2] = lm.z
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

    return data


def put_guide(frame, line1, line2="", color=(0, 255, 255)):
    """화면 상단에 안내 텍스트 표시"""
    cv2.putText(frame, line1, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if line2:
        cv2.putText(frame, line2, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)


def collect():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    print("=" * 50)
    print("  Dementia Prevention - Gesture Data Collection")
    print("  Press 'q' to quit")
    print("=" * 50)

    for g_idx, gesture in enumerate(GESTURES):
        # ── 동작 시작 전 대기 화면 ──
        print(f"\n> [{g_idx+1}/7] {gesture} - Get ready...")
        for countdown in range(90, 0, -1):  # 3초 대기 (30fps 기준)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            extract_landmarks(frame)  # 랜드마크 시각화만

            secs = countdown // 30 + 1
            put_guide(
                frame,
                f"[{g_idx+1}/7] {gesture}",
                f"Get ready... {secs}s",
                color=(0, 255, 255),
            )
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

        # ── 시퀀스 수집 ──
        for seq_idx in range(NUM_SEQUENCES):
            sequence = []

            for frame_idx in range(SEQUENCE_LEN):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                landmarks = extract_landmarks(frame)
                sequence.append(landmarks)

                # 진행 바
                progress = int((frame_idx + 1) / SEQUENCE_LEN * 200)
                cv2.rectangle(frame, (20, 110), (20 + progress, 130),
                              (0, 255, 0), -1)
                cv2.rectangle(frame, (20, 110), (220, 130),
                              (200, 200, 200), 2)

                put_guide(
                    frame,
                    f"{gesture}  [{seq_idx+1}/{NUM_SEQUENCES}]",
                    f"Recording: {frame_idx+1}/{SEQUENCE_LEN}",
                    color=(0, 255, 0),
                )
                cv2.imshow("Data Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # 시퀀스 저장
            save_path = os.path.join(DATA_PATH, gesture, f"{seq_idx}.npy")
            np.save(save_path, np.array(sequence))  # shape: (60, 126)

            if (seq_idx + 1) % 10 == 0:
                print(f"  {gesture}: {seq_idx+1}/{NUM_SEQUENCES} saved")

        print(f"[Done] {gesture}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n🎉 All data collected!")
    print(f"   Saved to: {DATA_PATH}/")


if __name__ == "__main__":
    collect()