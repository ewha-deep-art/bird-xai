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


def _collect_abs_coords(pred_delta, true_delta, last_obs, delta_scaler):
    """scaled Δ → 절대 좌표 (batch, horizon, 3)."""
    batch_size = pred_delta.shape[0]
    pred_abs, true_abs = [], []
    for b in range(batch_size):
        pred_d = delta_scaler.inverse_transform(pred_delta[b])
        true_d = delta_scaler.inverse_transform(true_delta[b])
        pred_abs.append(delta_to_absolute(last_obs[b], pred_d))
        true_abs.append(delta_to_absolute(last_obs[b], true_d))
    return np.stack(pred_abs), np.stack(true_abs)


@torch.no_grad()
def evaluate_lat_rmse(model, loader, device, delta_scaler):
    """전체 horizon 평균 lat RMSE (degree)."""
    model.eval()
    sq_err_sum = 0.0
    count = 0
    for X, y, last_obs in loader:
        pred_delta = model(X.to(device)).cpu().numpy()
        true_delta = y.numpy()
        last_np = last_obs.numpy()
        pred_abs, true_abs = _collect_abs_coords(pred_delta, true_delta, last_np, delta_scaler)
        diff = pred_abs[:, :, 0] - true_abs[:, :, 0]
        sq_err_sum += (diff ** 2).sum()
        count += diff.size
    return float(np.sqrt(sq_err_sum / count))


@torch.no_grad()
def persistence_lat_rmse(loader, delta_scaler):
    """Δ=0 persistence baseline의 lat RMSE (degree)."""
    zero_delta = np.zeros((FORECAST_HORIZON, len(TARGET_FEATURES)), dtype=np.float32)
    sq_err_sum = 0.0
    count = 0
    for _, y, last_obs in loader:
        true_delta = y.numpy()
        last_np = last_obs.numpy()
        batch_size = true_delta.shape[0]
        pred_delta = np.tile(zero_delta, (batch_size, 1, 1))
        _, true_abs = _collect_abs_coords(pred_delta, true_delta, last_np, delta_scaler)
        pred_lat = last_np[:, 0:1]
        diff = pred_lat - true_abs[:, :, 0]
        sq_err_sum += (diff ** 2).sum()
        count += diff.size
    return float(np.sqrt(sq_err_sum / count))


def test(model, test_loader, delta_scaler):
    test_loss = evaluate(model, test_loader, DEVICE)
    test_lat_rmse = evaluate_lat_rmse(model, test_loader, DEVICE, delta_scaler)
    persist_lat_rmse = persistence_lat_rmse(test_loader, delta_scaler)
    print(
        f"\n[최종] test MSE={test_loss:.4f}  "
        f"lat RMSE={test_lat_rmse:.4f}°  persistence={persist_lat_rmse:.4f}°  "
        f"beats_persist={test_lat_rmse < persist_lat_rmse}"
    )

    all_pred_abs, all_true_abs = [], []
    model.eval()
    with torch.no_grad():
        for X, y, last_obs in test_loader:
            pred_delta = model(X.to(DEVICE)).cpu().numpy()
            true_delta = y.numpy()
            last_np = last_obs.numpy()
            pred_abs, true_abs = _collect_abs_coords(pred_delta, true_delta, last_np, delta_scaler)
            all_pred_abs.append(pred_abs)
            all_true_abs.append(true_abs)

    pred_abs = np.concatenate(all_pred_abs, axis=0)
    true_abs = np.concatenate(all_true_abs, axis=0)

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
    delta_scaler = joblib.load(DELTA_SCALER_PATH)
    persist_val_lat = persistence_lat_rmse(val_loader, delta_scaler)
    print(f"[model] 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[baseline] val persistence lat RMSE={persist_val_lat:.4f}°")

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_lat_rmse = float('inf')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, DEVICE)
        val_lat_rmse = evaluate_lat_rmse(model, val_loader, DEVICE, delta_scaler)
        scheduler.step(val_lat_rmse)

        beat = "✓" if val_lat_rmse < persist_val_lat else " "
        print(
            f"[{epoch:03d}/{EPOCHS}] train={train_loss:.4f}  val={val_loss:.4f}  "
            f"val_lat={val_lat_rmse:.4f}°  persist={persist_val_lat:.4f}° {beat}  "
            f"({time.time()-t0:.1f}s)"
        )

        if val_lat_rmse < best_val_lat_rmse:
            best_val_lat_rmse = val_lat_rmse
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  → saved {MODEL_SAVE_PATH}  (val_lat={best_val_lat_rmse:.4f}°)")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    test(model, test_loader, delta_scaler)
