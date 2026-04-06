import torch
import torch.nn as nn
import numpy as np
import pickle
import os

# ── 설정 ──────────────────────────────────────────────────
SEQUENCE_LEN  = 30
NUM_KEYPOINTS = 42
USE_VELOCITY  = True
INPUT_SIZE    = NUM_KEYPOINTS * 3 * (2 if USE_VELOCITY else 1)  # 252
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
NUM_CLASSES  = 6

MODEL_DIR   = "/Users/jangjunseo/Desktop/dementiaProject/model"
MODEL_PATH  = os.path.join(MODEL_DIR, "gesture_final.pt")
ONNX_PATH   = os.path.join(MODEL_DIR, "gesture_final.onnx")
COREML_PATH = os.path.join(MODEL_DIR, "gesture_final.mlpackage")
LABEL_PATH  = os.path.join(MODEL_DIR, "label_encoder.pkl")


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
        out = out.mean(dim=1)  # mean pooling
        return self.fc(out)


# ── 전처리 포함 Wrapper 모델 ──────────────────────────────
class GestureModelWithPreprocess(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: (1, 30, 126) 원본 랜드마크 입력
        # 1. normalize_sequence
        B, T, F = x.shape
        x3 = x.reshape(B, T, -1, 3)
        min_xy = x3[:, :, :, :2].min(dim=2, keepdim=True).values
        max_xy = x3[:, :, :, :2].max(dim=2, keepdim=True).values
        x3[:, :, :, :2] = (x3[:, :, :, :2] - min_xy) / (max_xy - min_xy + 1e-6)
        x = x3.reshape(B, T, F)

        # 2. add_velocity (torch.diff 대신 슬라이싱 사용 - CoreML 호환)
        vel = x[:, 1:, :] - x[:, :-1, :]
        vel = torch.cat([vel, vel[:, -1:, :]], dim=1)
        x = torch.cat([x, vel], dim=2)  # (1, 30, 252)

        return self.model(x)


def convert():
    device = torch.device("cpu")  # 변환은 CPU에서

    # 1. 모델 로드
    model = GestureCNNLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[Loaded] 모델 → {MODEL_PATH}")

    # 레이블 로드
    with open(LABEL_PATH, "rb") as f:
        le = pickle.load(f)
    labels = list(le.classes_)
    print(f"[Labels] {labels}")

    # 2. PyTorch → ONNX 변환
    dummy_input = torch.randn(1, SEQUENCE_LEN, INPUT_SIZE)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"\n✅ ONNX 변환 완료 → {ONNX_PATH}")

    # 3. ONNX → Core ML 변환
    try:
        import coremltools as ct

        # 전처리 포함 wrapper로 변환
        wrapper = GestureModelWithPreprocess(model)
        wrapper.eval()
        example_input = torch.randn(1, SEQUENCE_LEN, 126)  # 원본 126 입력
        traced_model = torch.jit.trace(wrapper, example_input)

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=(1, SEQUENCE_LEN, 126))],  # 126 입력
            minimum_deployment_target=ct.target.iOS15,
            source="pytorch",
        )

        # 메타데이터 설정
        mlmodel.short_description = "Dementia Prevention Gesture Recognition"
        mlmodel.save(COREML_PATH)
        print(f"✅ Core ML 변환 완료 → {COREML_PATH}")

    except ImportError:
        print("\n⚠️  coremltools 미설치")
        print("   pip install coremltools")
        print(f"\n   ONNX 파일은 저장됨 → {ONNX_PATH}")
        print("   설치 후 다시 실행하거나 ONNX를 직접 Expo에서 사용 가능")

    # 4. 레이블 파일 저장 (Expo에서 사용)
    labels_path = os.path.join(MODEL_DIR, "labels.json")
    import json
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({
            "labels": labels,
            "labels_ko": {
                "finger_wave":    "손가락 움직이기",
                "hand_shake":     "손 털기",
                "finger_fold":    "손가락 접기",
                "fist_open":      "주먹 쥐고 펴기",
                "cross_fist":     "엇갈려 주먹 쥐고 펴기",
                "fingertip_clap": "손끝 박수",
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ 레이블 JSON 저장 → {labels_path}")

    print("\n" + "="*50)
    print("  다음 단계 (Expo)")
    print("="*50)
    print("1. gesture_final.mlpackage → iOS 프로젝트에 추가")
    print("2. labels.json → assets에 추가")
    print("3. expo-camera로 프레임 캡처")
    print("4. MediaPipe JS로 랜드마크 추출")
    print("5. Core ML 모델로 예측")


if __name__ == "__main__":
    convert()