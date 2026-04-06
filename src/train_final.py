import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import pickle

# ── 랜덤 시드 고정 ─────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

# ── 설정 ──────────────────────────────────────────────────
GESTURES = [
    "finger_wave",
    "hand_shake",
    "finger_fold",
    "fist_open",
    "cross_fist",
    "fingertip_clap",
]
SEQUENCE_LEN  = 30
NUM_KEYPOINTS = 42
USE_VELOCITY  = True
INPUT_SIZE    = NUM_KEYPOINTS * 3 * (2 if USE_VELOCITY else 1)  # 252

HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
NUM_CLASSES  = len(GESTURES)
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
EARLY_STOP   = 15

DATA_RANGE = (0, 799)  # 전체 데이터

DATA_PATH  = "/Users/jangjunseo/Desktop/dementiaProject/data/sequences"
MODEL_DIR  = "/Users/jangjunseo/Desktop/dementiaProject/model"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_final.pt")
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── 전처리 ────────────────────────────────────────────────
def normalize_sequence(seq):
    """bounding box 기반 정규화"""
    seq = seq.reshape(SEQUENCE_LEN, -1, 3)
    min_xy = seq[:, :, :2].min(axis=1, keepdims=True)
    max_xy = seq[:, :, :2].max(axis=1, keepdims=True)
    seq[:, :, :2] = (seq[:, :, :2] - min_xy) / (max_xy - min_xy + 1e-6)
    return seq.reshape(SEQUENCE_LEN, -1)


def add_velocity(seq):
    """velocity feature 추가"""
    vel = np.diff(seq, axis=0)
    vel = np.vstack([vel, vel[-1]])
    return np.concatenate([seq, vel], axis=1)


# ── 데이터 증강 ───────────────────────────────────────────
def augment_sequence(seq):
    """가우시안 노이즈 + 스케일링"""
    seq += np.random.normal(0, 0.01, seq.shape)
    scale = np.random.uniform(0.8, 1.2)
    seq *= scale
    return seq


# ── Dataset ───────────────────────────────────────────────
class GestureDataset(Dataset):
    def __init__(self, data_path, gestures, file_range, augment=False):
        self.X, self.y = [], []
        labels = []
        start, end = file_range

        for g in gestures:
            folder = os.path.join(data_path, g)
            if not os.path.exists(folder):
                print(f"[WARN] 폴더 없음: {folder} → 스킵")
                continue
            for idx in range(start, end + 1):
                fpath = os.path.join(folder, f"{idx}.npy")
                if not os.path.exists(fpath):
                    continue
                seq = np.load(fpath)
                if seq.shape != (SEQUENCE_LEN, NUM_KEYPOINTS * 3):
                    continue

                seq = normalize_sequence(seq)
                if USE_VELOCITY:
                    seq = add_velocity(seq)

                # 원본 추가
                self.X.append(seq.copy())
                labels.append(g)

                # 증강 추가 (원본과 별도로)
                if augment:
                    self.X.append(augment_sequence(seq.copy()))
                    labels.append(g)

        self.le = LabelEncoder()
        self.le.fit(gestures)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = self.le.transform(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


# ── 학습 ──────────────────────────────────────────────────
def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device} 사용")
    print(f"[Model] CNN-LSTM (정규화 + Velocity + Mean Pooling)\n")

    dataset = GestureDataset(DATA_PATH, GESTURES, DATA_RANGE, augment=True)
    print(f"[Dataset] 총 {len(dataset.X)}개 로드 (증강 포함)")

    if len(dataset.X) == 0:
        print("❌ 데이터가 없습니다.")
        return

    with open(LABEL_PATH, "wb") as f:
        pickle.dump(dataset.le, f)
    print(f"[Saved] 레이블 인코더 → {LABEL_PATH}")

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(dataset.X, dtype=torch.float32),
            torch.tensor(dataset.y, dtype=torch.long)
        ), batch_size=BATCH_SIZE, shuffle=True
    )

    model     = GestureCNNLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss  = float('inf')
    no_improve = 0

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'':>15}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = total = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out  = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(dim=1) == y_b).sum().item()
            total      += len(y_b)

        avg_loss  = total_loss / len(train_loader)
        train_acc = correct / total

        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            status = "✅ saved"
        else:
            no_improve += 1
            status = f"no improve ({no_improve}/{EARLY_STOP})"

        print(f"{epoch:>6} | {avg_loss:>10.4f} | {train_acc:>9.1%} | {status}")

        if no_improve >= EARLY_STOP:
            print(f"\n⏹ Early Stop at epoch {epoch}")
            break

        scheduler.step()

    print(f"\n✅ 학습 완료!")
    print(f"   Best train loss : {best_loss:.4f}")
    print(f"   모델 저장 위치  : {MODEL_PATH}")
    print(f"\n→ 다음 단계: python convert_coreml.py")


if __name__ == "__main__":
    train()