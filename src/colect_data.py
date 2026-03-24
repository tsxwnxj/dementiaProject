import os
import cv2
import numpy as np
import mediapipe as mp
import argparse

# ── 설정 ──────────────────────────────────────────────────
ALL_GESTURES = {
    "1": "finger_wave",     # 1. 손가락 움직이기
    "2": "hand_shake",      # 2. 손털기
    "3": "finger_fold",     # 3. 손가락 접기
    "4": "fist_open",       # 4. 주먹 쥐고 펴기
    "5": "cross_fist",      # 5. 엇갈려 주먹 쥐고 펴기
    "6": "fingertip_clap",  # 6. 손끝 박수
}
SEQUENCE_LEN  = 30   # 동작 1회 = 30프레임 (~1초)
NUM_SEQUENCES = 50   # 동작당 추가 수집할 시퀀스 수
DATA_PATH     = r"C:\Users\jaemi\dementiaProject3\data\sequences"

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
for g in ALL_GESTURES.values():
    os.makedirs(os.path.join(DATA_PATH, g), exist_ok=True)


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


def put_guide(frame, line1, line2="", color=(0, 255, 255)):
    cv2.putText(frame, line1, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if line2:
        cv2.putText(frame, line2, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)


def get_start_idx(gesture):
    folder = os.path.join(DATA_PATH, gesture)
    existing = [f for f in os.listdir(folder) if f.endswith(".npy")]
    return len(existing)


def select_gestures():
    """대화형으로 수집할 동작 선택"""
    print("\n" + "=" * 50)
    print("  Select gestures to collect:")
    print("=" * 50)
    for num, name in ALL_GESTURES.items():
        folder = os.path.join(DATA_PATH, name)
        existing = len([f for f in os.listdir(folder) if f.endswith(".npy")])
        print(f"  {num}. {name:<20} (existing: {existing})")
    print("=" * 50)
    print("  Enter numbers to collect (e.g. 1 2 3 or 'all')")
    print("=" * 50)

    user_input = input("  > ").strip().lower()

    if user_input == "all":
        return list(ALL_GESTURES.values())

    selected = []
    for num in user_input.split():
        if num in ALL_GESTURES:
            selected.append(ALL_GESTURES[num])
        else:
            print(f"  [WARN] '{num}' is invalid, skipping")

    return selected


def collect(gestures):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    print("\n" + "=" * 50)
    print("  Dementia Prevention - Gesture Data Collection")
    print(f"  Collecting: {', '.join(gestures)}")
    print("  Press 'q' to quit")
    print("=" * 50)

    for g_idx, gesture in enumerate(gestures):
        start_idx = get_start_idx(gesture)
        print(f"\n> [{g_idx+1}/{len(gestures)}] {gesture} - existing: {start_idx}, new: {NUM_SEQUENCES}")

        # ── 동작 시작 전 대기 화면 ──
        for countdown in range(90, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            extract_landmarks(frame)

            secs = countdown // 30 + 1
            put_guide(
                frame,
                f"[{g_idx+1}/{len(gestures)}] {gesture}",
                f"Get ready... {secs}s  (existing: {start_idx})",
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
            save_idx = start_idx + seq_idx

            for frame_idx in range(SEQUENCE_LEN):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                landmarks = extract_landmarks(frame)
                sequence.append(landmarks)

                progress = int((frame_idx + 1) / SEQUENCE_LEN * 200)
                cv2.rectangle(frame, (20, 110), (20 + progress, 130),
                              (0, 255, 0), -1)
                cv2.rectangle(frame, (20, 110), (220, 130),
                              (200, 200, 200), 2)

                put_guide(
                    frame,
                    f"{gesture}  [{seq_idx+1}/{NUM_SEQUENCES}]  (#{save_idx})",
                    f"Recording: {frame_idx+1}/{SEQUENCE_LEN}",
                    color=(0, 255, 0),
                )
                cv2.imshow("Data Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            save_path = os.path.join(DATA_PATH, gesture, f"{save_idx}.npy")
            np.save(save_path, np.array(sequence))

            if (seq_idx + 1) % 10 == 0:
                print(f"  {gesture}: {seq_idx+1}/{NUM_SEQUENCES} saved (#{save_idx})")

        print(f"[Done] {gesture} (total: {start_idx + NUM_SEQUENCES})")

    cap.release()
    cv2.destroyAllWindows()
    print("\n🎉 All data collected!")
    print(f"   Saved to: {DATA_PATH}/")


if __name__ == "__main__":
    # 커맨드라인 인자로도 선택 가능
    # 예: python collect_data.py --gestures 1 4 5
    parser = argparse.ArgumentParser()
    parser.add_argument("--gestures", nargs="*", help="Gesture numbers (e.g. 1 4 5)")
    args = parser.parse_args()

    if args.gestures:
        selected = [ALL_GESTURES[n] for n in args.gestures if n in ALL_GESTURES]
    else:
        selected = select_gestures()

    if not selected:
        print("❌ No gestures selected.")
    else:
        collect(selected)