import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
import numpy as np
import time

from ai.common import (
    MODEL_SAVE_PATH, DELTA_SCALER_PATH, TARGET_FEATURES, DEVICE,
    FORECAST_HORIZON, train_loader, val_loader, test_loader,
    delta_to_absolute,
)
from ai.training.model import load_model

# ─── 하이퍼파라미터 ────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0023630238233190467
TARGET_LOSS_WEIGHTS = torch.tensor([1.0, 1.0, 0.1])  # lat, lon, height_raw

# ─── 학습 ────────────────────────────────────────────────────────────────────

def weighted_mse_loss(pred, target):
    weights = TARGET_LOSS_WEIGHTS.to(pred.device)
    return (weights * (pred - target) ** 2).mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y, _ in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = weighted_mse_loss(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for X, y, _ in loader:
        X, y = X.to(device), y.to(device)
        total_loss += weighted_mse_loss(model(X), y).item()
    return total_loss / len(loader)


def test(model, test_loader):
    test_loss = evaluate(model, test_loader, DEVICE)
    print(f"\n[최종] test MSE={test_loss:.4f}  RMSE={test_loss**0.5:.4f}")

    delta_scaler = joblib.load(DELTA_SCALER_PATH)

    all_pred_abs, all_true_abs = [], []
    model.eval()
    with torch.no_grad():
        for X, y, last_obs in test_loader:
            pred_delta = model(X.to(DEVICE)).cpu().numpy()
            true_delta = y.numpy()
            last_np = last_obs.numpy()

            batch_size = pred_delta.shape[0]
            for b in range(batch_size):
                pred_d = delta_scaler.inverse_transform(pred_delta[b].reshape(-1, len(TARGET_FEATURES)))
                true_d = delta_scaler.inverse_transform(true_delta[b].reshape(-1, len(TARGET_FEATURES)))
                all_pred_abs.append(delta_to_absolute(last_np[b], pred_d))
                all_true_abs.append(delta_to_absolute(last_np[b], true_d))

    pred_abs = np.stack(all_pred_abs)   # (N, M, 3)
    true_abs = np.stack(all_true_abs)

    rmse_per_target = np.sqrt(((pred_abs - true_abs) ** 2).mean(axis=(0, 1)))
    mae_per_target = np.abs(pred_abs - true_abs).mean(axis=(0, 1))

    print("\n[실제 단위 오차 — 전체 horizon 평균]")
    print(f"  {'target':<12} {'RMSE':>10} {'MAE':>10}")
    print(f"  {'-'*34}")
    for name, rmse, mae in zip(TARGET_FEATURES, rmse_per_target, mae_per_target):
        print(f"  {name:<12} {rmse:>10.4f} {mae:>10.4f}")

    print("\n[horizon별 lat RMSE (degree)]")
    for step in (0, 5, FORECAST_HORIZON - 1):
        rmse = np.sqrt(((pred_abs[:, step, 0] - true_abs[:, step, 0]) ** 2).mean())
        print(f"  step {step + 1:2d}: {rmse:.4f}")


def main():
    model = load_model()
    print(f"[model] 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_loss)

        print(f"[{epoch:03d}/{EPOCHS}] train={train_loss:.4f}  val={val_loss:.4f}  ({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  → saved {MODEL_SAVE_PATH}  (val={best_val_loss:.4f})")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    test(model, test_loader)
