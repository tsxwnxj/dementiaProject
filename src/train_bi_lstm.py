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
SEQUENCE_LEN = 30
INPUT_SIZE   = 126
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
NUM_CLASSES  = len(GESTURES)
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
EARLY_STOP   = 15

TRAIN_RANGE = (0, 399)
VAL_RANGE   = (400, 499)

DATA_PATH  = "/Users/jangjunseo/Desktop/dementiaProject/data/sequences"
MODEL_DIR  = "/Users/jangjunseo/Desktop/dementiaProject/model"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_bi_lstm.pt")
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────
class GestureDataset(Dataset):
    def __init__(self, data_path, gestures, file_range):
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
                if seq.shape != (SEQUENCE_LEN, INPUT_SIZE):
                    continue
                self.X.append(seq)
                labels.append(g)

        self.le = LabelEncoder()
        self.le.fit(gestures)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = self.le.transform(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Bi-LSTM 모델 ───────────────────────────────────────────
class GestureBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,   # ← 이것만 추가
        )
        # bidirectional이라 hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])  # 마지막 타임스텝


# ── 학습 ──────────────────────────────────────────────────
def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device} 사용")
    print(f"[Model] Bi-LSTM\n")

    train_dataset = GestureDataset(DATA_PATH, GESTURES, TRAIN_RANGE)
    val_dataset   = GestureDataset(DATA_PATH, GESTURES, VAL_RANGE)

    print(f"[Train] {len(train_dataset.X)}개 (파일 {TRAIN_RANGE[0]}~{TRAIN_RANGE[1]})")
    print(f"[Val]   {len(val_dataset.X)}개   (파일 {VAL_RANGE[0]}~{VAL_RANGE[1]})\n")

    if len(train_dataset.X) == 0 or len(val_dataset.X) == 0:
        print("❌ 데이터가 없습니다.")
        return

    with open(LABEL_PATH, "wb") as f:
        pickle.dump(train_dataset.le, f)

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

    model     = GestureBiLSTM().to(device)
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

    print(f"\n✅ 학습 완료! [Bi-LSTM]")
    print(f"   Best val accuracy : {best_val_acc:.1%}")
    print(f"   모델 저장 위치    : {MODEL_PATH}")

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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Bi-LSTM Results (Best Val Acc: {best_val_acc:.1%})', fontsize=14)
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
    save_path = os.path.join(MODEL_DIR, "bi_lstm_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"그래프 저장 → {save_path}")


if __name__ == "__main__":
    train()