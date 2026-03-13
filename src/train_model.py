import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
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
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
NUM_CLASSES  = len(GESTURES)
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
NUM_MODELS   = 3
NOISE_LEVELS = [0.005, 0.01]

DATA_PATH  = "/Users/jangjunseo/Desktop/dementiaProject/data/sequences"
MODEL_DIR  = "/Users/jangjunseo/Desktop/dementiaProject/model"
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── 데이터 증강 ───────────────────────────────────────────
def augment(sequence, noise_std):
    noise = np.random.normal(0, noise_std, sequence.shape).astype(np.float32)
    return sequence + noise


# ── 원본 Dataset (증강 없음) ───────────────────────────────
class RawGestureDataset(Dataset):
    def __init__(self, data_path, gestures):
        self.X, self.y = [], []
        labels = []
        self.le = LabelEncoder()

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
                seq = np.load(os.path.join(folder, fname))
                if seq.shape != (SEQUENCE_LEN, INPUT_SIZE):
                    print(f"[WARN] shape 불일치 {seq.shape} → 스킵: {fname}")
                    continue
                self.X.append(seq)
                labels.append(g)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = self.le.fit_transform(labels)
        print(f"\n[Dataset] 원본 총 {len(self.X)}개 로드")
        print(f"[Dataset] 클래스: {list(self.le.classes_)}\n")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# ── 증강 Dataset (train 전용) ──────────────────────────────
class AugmentedDataset(Dataset):
    def __init__(self, X, y):
        """원본 X, y를 받아서 증강 포함한 Dataset 생성"""
        aug_X, aug_y = list(X), list(y)

        for seq, label in zip(X, y):
            for noise_std in NOISE_LEVELS:
                aug_X.append(augment(seq, noise_std))
                aug_y.append(label)

        self.X = torch.tensor(np.array(aug_X), dtype=torch.float32)
        self.y = torch.tensor(np.array(aug_y), dtype=torch.long)
        print(f"[Augmented] 원본 {len(X)}개 → 증강 후 {len(self.X)}개")

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

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


# ── 단일 모델 학습 ─────────────────────────────────────────
def train_single(model_idx, train_loader, val_loader, device):
    model     = GestureLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )

    best_val_acc = 0.0
    model_path   = os.path.join(MODEL_DIR, f"gesture_lstm_{model_idx}.pt")

    print(f"\n{'─'*40}")
    print(f"  Model {model_idx+1}/{NUM_MODELS} 학습 시작")
    print(f"{'─'*40}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Acc':>8}")
    print("-" * 32)

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

        print(f"{epoch:>6} | {avg_loss:>10.4f} | {val_acc:>7.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

        scheduler.step()

    print(f"\n  ✅ Model {model_idx+1} 완료 | Best val acc: {best_val_acc:.1%}")
    return model_path, best_val_acc


# ── 앙상블 평가 ───────────────────────────────────────────
def evaluate_ensemble(model_paths, val_loader, device, label_classes):
    models = []
    for path in model_paths:
        m = GestureLSTM().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device)
            probs = torch.zeros(X_b.size(0), NUM_CLASSES).to(device)
            for m in models:
                probs += torch.softmax(m(X_b), dim=1)
            probs /= len(models)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.numpy())

    ensemble_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n{'='*40}")
    print(f"  🎯 앙상블 Val Accuracy: {ensemble_acc:.1%}")
    print(f"{'='*40}\n")

    print(classification_report(all_labels, all_preds, target_names=label_classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=label_classes,
                yticklabels=label_classes,
                cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Ensemble Confusion Matrix (acc: {ensemble_acc:.1%})')
    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"혼동 행렬 저장 → {save_path}")
    return ensemble_acc


# ── 메인 ──────────────────────────────────────────────────
def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device} 사용")

    # 1. 원본 데이터 로드
    raw_dataset = RawGestureDataset(DATA_PATH, GESTURES)
    if len(raw_dataset) == 0:
        print("❌ 데이터가 없습니다.")
        return

    # 레이블 인코더 저장
    with open(LABEL_PATH, "wb") as f:
        pickle.dump(raw_dataset.le, f)
    print(f"[Saved] 레이블 인코더 → {LABEL_PATH}")

    # 2. 원본 기준으로 train/val 분리 (누수 방지)
    indices = list(range(len(raw_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED,
        stratify=raw_dataset.y  # 클래스 비율 유지
    )

    X_train = raw_dataset.X[train_idx]
    y_train = raw_dataset.y[train_idx]
    X_val   = raw_dataset.X[val_idx]
    y_val   = raw_dataset.y[val_idx]

    print(f"[Split] 원본 train: {len(X_train)}개 / val: {len(X_val)}개")

    # 3. train에만 증강 적용
    train_dataset = AugmentedDataset(X_train, y_train)

    # val은 원본 그대로
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    print(f"[Augmented] 최종 train: {len(train_dataset)}개 / val: {len(val_dataset)}개\n")

    # 4. 앙상블 학습
    model_paths, best_accs = [], []
    for i in range(NUM_MODELS):
        torch.manual_seed(SEED + i)
        path, acc = train_single(i, train_loader, val_loader, device)
        model_paths.append(path)
        best_accs.append(acc)

    print(f"\n{'='*40}")
    print(f"  개별 모델 성능:")
    for i, acc in enumerate(best_accs):
        print(f"  Model {i+1}: {acc:.1%}")

    # 5. 앙상블 평가
    evaluate_ensemble(model_paths, val_loader, device, raw_dataset.le.classes_)

    # 6. ensemble_info 저장
    with open(os.path.join(MODEL_DIR, "ensemble_info.pkl"), "wb") as f:
        pickle.dump({"model_paths": model_paths, "num_models": NUM_MODELS}, f)
    print(f"[Saved] 앙상블 정보 → {MODEL_DIR}/ensemble_info.pkl")


if __name__ == "__main__":
    train()