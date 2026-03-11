import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import pickle

# ── 설정 ──────────────────────────────────────────────────
GESTURES = [
    "finger_wave",     # 1. 손가락 움직이기
    "hand_shake",      # 2. 손털기
    "finger_fold",     # 3. 손가락 접기
    "fist_open",       # 4. 주먹 쥐고 펴기
    "cross_fist",      # 5. 엇갈려 주먹 쥐고 펴기
    "fingertip_clap",  # 6. 손끝 박수
]
SEQUENCE_LEN = 30    # collect_data.py 와 반드시 일치
INPUT_SIZE   = 126   # 양손 21개 관절 × xyz × 2
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
NUM_CLASSES  = len(GESTURES)
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
DATA_PATH    = "data/sequences"
MODEL_PATH   = "model/gesture_lstm.pt"
LABEL_PATH   = "model/label_encoder.pkl"  # 예측 시 재사용

os.makedirs("model", exist_ok=True)


# ── Dataset ───────────────────────────────────────────────
class GestureSeqDataset(Dataset):
    def __init__(self, data_path, gestures):
        self.X, self.y = [], []
        labels = []

        for g in gestures:
            folder = os.path.join(data_path, g)
            if not os.path.exists(folder):
                print(f"[WARN] 폴더 없음: {folder} → 스킵")
                continue
            files = [f for f in os.listdir(folder) if f.endswith(".npy")]
            if len(files) == 0:
                print(f"[WARN] 데이터 없음: {folder} → 스킵")
                continue
            for fname in files:
                seq = np.load(os.path.join(folder, fname))  # (60, 126)
                if seq.shape == (SEQUENCE_LEN, INPUT_SIZE):
                    self.X.append(seq)
                    labels.append(g)
                else:
                    print(f"[WARN] shape 불일치 {seq.shape} → 스킵: {fname}")

        self.le = LabelEncoder()
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(
            self.le.fit_transform(labels), dtype=torch.long
        )
        print(f"\n[Dataset] 총 {len(self.X)}개 시퀀스 로드")
        print(f"[Dataset] 클래스: {list(self.le.classes_)}\n")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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

    def forward(self, x):          # x: (B, 60, 126)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])  # 마지막 타임스텝


# ── 학습 함수 ─────────────────────────────────────────────
def train():
    # 디바이스 설정 (Mac M1/M2/M3 → MPS 자동 사용)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device} 사용\n")

    # 데이터 로드
    dataset = GestureSeqDataset(DATA_PATH, GESTURES)
    if len(dataset) == 0:
        print("❌ 데이터가 없습니다. collect_data.py 먼저 실행하세요.")
        return

    # 레이블 인코더 저장 (predict.py에서 재사용)
    with open(LABEL_PATH, "wb") as f:
        pickle.dump(dataset.le, f)
    print(f"[Saved] 레이블 인코더 → {LABEL_PATH}")

    # train / val 분리 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    print(f"[Split] train: {train_size}개 / val: {val_size}개\n")

    # 모델 초기화
    model     = GestureLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )

    best_val_acc = 0.0

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Acc':>8}")
    print("-" * 32)

    for epoch in range(1, EPOCHS + 1):
        # ── 학습 ──
        model.train()
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ── 검증 ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                preds = model(X_b.to(device)).argmax(dim=1)
                correct += (preds == y_b.to(device)).sum().item()
                total   += len(y_b)

        avg_loss = total_loss / len(train_loader)
        val_acc  = correct / total if total > 0 else 0

        print(f"{epoch:>6} | {avg_loss:>10.4f} | {val_acc:>7.1%}")

        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

        scheduler.step()

    print(f"\n✅ 학습 완료!")
    print(f"   Best val accuracy : {best_val_acc:.1%}")
    print(f"   모델 저장 위치    : {MODEL_PATH}")


if __name__ == "__main__":
    train()