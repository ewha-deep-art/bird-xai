import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time

from ai.common import (
    MODEL_DIR, DEVICE, BIRD,
    train_loader, val_loader
)
from ai.training.model import load_model


# ─── 하이퍼파라미터 ────────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 0.008722902191551666

# ─── 학습 ────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total_loss += criterion(model(X), y).item()
    return total_loss / len(loader)

def train():
    model = load_model()
    print(f"[model] 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss  = float('inf')
    model_save_path = MODEL_DIR / f"{BIRD}_best.pt"

    for epoch in range(1, EPOCHS + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        print(f"[{epoch:03d}/{EPOCHS}] train={train_loss:.4f}  val={val_loss:.4f}  ({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  → saved {model_save_path}  (val={best_val_loss:.4f})")

if __name__ == "__main__":
    train()