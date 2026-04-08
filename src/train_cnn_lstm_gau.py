import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
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
NUM_KEYPOINTS = 42        # 양손 21개 × 2
INPUT_SIZE    = NUM_KEYPOINTS * 3  # 126 (x,y,z)
USE_VELOCITY  = True
if USE_VELOCITY:
    INPUT_SIZE *= 2       # 126 → 252 (position + velocity)

HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
NUM_CLASSES  = len(GESTURES)
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
EARLY_STOP   = 15

TRAIN_RANGE = (0, 699)   # 7명 학습
VAL_RANGE   = (700, 799) # 8번째 사람 검증

DATA_PATH  = "/Users/jangjunseo/Desktop/dementiaProject/data/sequences"
MODEL_DIR  = "/Users/jangjunseo/Desktop/dementiaProject/model"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_cnn_lstm.pt")
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── 전처리 ────────────────────────────────────────────────
def normalize_sequence(seq):
    """bounding box 기반 정규화 → 카메라 거리/위치 무관"""
    seq = seq.reshape(SEQUENCE_LEN, -1, 3)
    min_xy = seq[:, :, :2].min(axis=1, keepdims=True)
    max_xy = seq[:, :, :2].max(axis=1, keepdims=True)
    seq[:, :, :2] = (seq[:, :, :2] - min_xy) / (max_xy - min_xy + 1e-6)
    return seq.reshape(SEQUENCE_LEN, -1)


def add_velocity(seq):
    """velocity feature 추가 → 동작 방향/속도 정보"""
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
                    print(f"[WARN] shape 불일치 {seq.shape} → 스킵: {fpath}")
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

    # 1. 데이터 로드
    train_dataset = GestureDataset(DATA_PATH, GESTURES, TRAIN_RANGE, augment=True)
    val_dataset   = GestureDataset(DATA_PATH, GESTURES, VAL_RANGE,   augment=False)

    print(f"[Train] {len(train_dataset.X)}개 (파일 {TRAIN_RANGE[0]}~{TRAIN_RANGE[1]})")
    print(f"[Val]   {len(val_dataset.X)}개   (파일 {VAL_RANGE[0]}~{VAL_RANGE[1]}) ← 원본 그대로\n")

    if len(train_dataset.X) == 0 or len(val_dataset.X) == 0:
        print("❌ 데이터가 없습니다.")
        return

    with open(LABEL_PATH, "wb") as f:
        pickle.dump(train_dataset.le, f)
    print(f"[Saved] 레이블 인코더 → {LABEL_PATH}")

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_dataset.X, dtype=torch.float32),
            torch.tensor(train_dataset.y, dtype=torch.long)
        ), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_dataset.X, dtype=torch.float32),
            torch.tensor(val_dataset.y, dtype=torch.long)
        ), batch_size=BATCH_SIZE
    )

    # 2. 모델 학습
    model     = GestureCNNLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc     = 0.0
    no_improve       = 0
    early_stop_epoch = None
    history_loss     = []
    history_val_acc  = []

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Acc':>8} | {'':>15}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                preds = model(X_b.to(device)).argmax(dim=1)
                correct += (preds == y_b.to(device)).sum().item()
                total   += len(y_b)

        avg_loss = total_loss / len(train_loader)
        val_acc  = correct / total if total > 0 else 0

        history_loss.append(avg_loss)
        history_val_acc.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), MODEL_PATH)
            status = "✅ saved"
        else:
            no_improve += 1
            status = f"no improve ({no_improve}/{EARLY_STOP})"

        print(f"{epoch:>6} | {avg_loss:>10.4f} | {val_acc:>7.1%} | {status}")

        if no_improve >= EARLY_STOP:
            early_stop_epoch = epoch
            print(f"\n⏹ Early Stop at epoch {epoch}")
            break

        scheduler.step()

    print(f"\n✅ 학습 완료! [CNN-LSTM]")
    print(f"   Best val accuracy : {best_val_acc:.1%}")
    print(f"   모델 저장 위치    : {MODEL_PATH}")

    # 3. 평가
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            preds = model(X_b.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.numpy())

    print("\n" + classification_report(
        all_labels, all_preds,
        target_names=train_dataset.le.classes_
    ))

    # 4. 그래프 (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'CNN-LSTM Results (Best Val Acc: {best_val_acc:.1%})', fontsize=14)
    epochs_range = range(1, len(history_loss) + 1)

    axes[0, 0].plot(epochs_range, history_loss, 'b-o', markersize=3, label='Train Loss')
    if early_stop_epoch:
        axes[0, 0].axvline(x=early_stop_epoch, color='r', linestyle='--', label='Early Stop')
    axes[0, 0].set_title('Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs_range, [a * 100 for a in history_val_acc],
                    'g-o', markersize=3, label='Val Accuracy')
    axes[0, 1].axhline(y=best_val_acc * 100, color='r', linestyle='--',
                       label=f'Best: {best_val_acc:.1%}')
    if early_stop_epoch:
        axes[0, 1].axvline(x=early_stop_epoch, color='orange', linestyle='--', label='Early Stop')
    axes[0, 1].set_title('Val Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=train_dataset.le.classes_,
                yticklabels=train_dataset.le.classes_,
                cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].tick_params(axis='x', rotation=45)

    sample_count = len(all_labels)
    axes[1, 1].scatter(range(sample_count), all_labels,
                       c='blue', s=5, alpha=0.5, label='Actual')
    axes[1, 1].scatter(range(sample_count), all_preds,
                       c='red', s=5, alpha=0.5, label='Predicted')
    axes[1, 1].set_title('Actual vs Predicted (all samples)')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Class')
    axes[1, 1].set_yticks(range(NUM_CLASSES))
    axes[1, 1].set_yticklabels(train_dataset.le.classes_, fontsize=8)
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, "cnn_lstm_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"그래프 저장 → {save_path}")


if __name__ == "__main__":
    train()