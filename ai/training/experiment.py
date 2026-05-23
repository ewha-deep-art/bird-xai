import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import optuna
from optuna.trial import Trial

import time
import argparse
from pathlib import Path
import joblib

from data.loader import get_data_loader
from ai.common import DEVICE, ALL_FEATURES, TARGET_FEATURES, TARGET_SCALER_PATH
from ai.training.model import BirdLSTM
from ai.training.train import train_one_epoch, evaluate

# ─── 경로 ────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ─── Optuna objective ─────────────────────────────────────────────────────────

def make_objective(bird: str, features: list, args):
    """클로저로 데이터/고정 설정을 캡처한 objective 함수 반환."""
    # 데이터는 trial마다 재로드하지 않도록 미리 로드
    train_loader, val_loader, _ = get_data_loader(
        bird=bird,
        features=features,
        window_size=args.window_size,
        batch_size=args.batch_size,
    )

    def objective(trial: Trial) -> float:
        # ── 탐색 공간 정의 ──────────────────────────────────────────────────
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        num_layers  = trial.suggest_int("num_layers", 1, 3)
        dropout     = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # ── 모델 초기화 ─────────────────────────────────────────────────────
        model = BirdLSTM(
            input_size=len(features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=len(TARGET_FEATURES),
            dropout=dropout,
        ).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        best_val_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # 중간 결과 보고 → 성능 나쁜 trial 조기 종료
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_loss

    return objective


# ─── 최적 파라미터로 최종 학습 ────────────────────────────────────────────────

def train_best(bird: str, features: list, best_params: dict, args):
    """study에서 찾은 최적 파라미터로 재학습 후 test 평가."""

    train_loader, val_loader, test_loader = get_data_loader(
        bird=bird,
        features=features,
        window_size=args.window_size,
        batch_size=args.batch_size,
    )

    model = BirdLSTM(
        input_size=len(features),
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        output_size=len(TARGET_FEATURES),
        dropout=best_params["dropout"],
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=best_params["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    checkpoint_path = CHECKPOINT_DIR / f"{bird}_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        print(f"[{epoch:03d}/{args.epochs}] train={train_loss:.4f}  val={val_loss:.4f}  ({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → checkpoint saved (val={best_val_loss:.4f})")

    # 테스트
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test_loss = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n[최종] test MSE={test_loss:.4f}  RMSE={test_loss**0.5:.4f}")

    # ── 역변환 후 실제 단위 오차 계산 ──────────────────────────────
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    all_preds, all_trues = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X.to(DEVICE)).cpu().numpy()  # (batch, ws, n_targets)
            true = y.numpy()
            all_preds.append(pred)
            all_trues.append(true)

    # (N, ws, n_targets) → (N*ws, n_targets) 로 펼쳐서 역변환
    preds = np.concatenate(all_preds).reshape(-1, len(TARGET_FEATURES))
    trues = np.concatenate(all_trues).reshape(-1, len(TARGET_FEATURES))

    preds_real = target_scaler.inverse_transform(preds)
    trues_real = target_scaler.inverse_transform(trues)

    rmse_per_target = np.sqrt(((preds_real - trues_real) ** 2).mean(axis=0))
    mae_per_target  = np.abs(preds_real - trues_real).mean(axis=0)

    print("\n[실제 단위 오차]")
    print(f"  {'target':<12} {'RMSE':>10} {'MAE':>10}")
    print(f"  {'-'*34}")
    for name, rmse, mae in zip(TARGET_FEATURES, rmse_per_target, mae_per_target):
        print(f"  {name:<12} {rmse:>10.4f} {mae:>10.4f}")

# ─── 진입점 ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Bird LSTM + Optuna Experiment")

    # 데이터
    parser.add_argument("--bird",        type=str, required=True)
    parser.add_argument("--features",    type=str, nargs="+", help="기본: ALL_FEATURES")
    parser.add_argument("--window_size", type=int, default=24)
    parser.add_argument("--batch_size",  type=int, default=32)

    # 학습
    parser.add_argument("--epochs",      type=int, default=50)

    # Optuna
    parser.add_argument("--n_trials",    type=int, default=30,  help="탐색할 trial 수")
    parser.add_argument("--study_name",  type=str, default=None, help="study 이름 (기본: bird명)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    features = args.features or ALL_FEATURES
    study_name = args.study_name or f"bird_lstm_{args.bird}"

    # pruner: 성능 나쁜 trial을 epoch 중간에 조기 종료
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study  = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=pruner,
    )

    print(f"[Optuna] study='{study_name}'  n_trials={args.n_trials}")
    study.optimize(
        make_objective(args.bird, features, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # 결과 출력
    best = study.best_trial
    print(f"\n[탐색 완료] best trial #{best.number}  val_loss={best.value:.4f}")
    print(f"  파라미터: {best.params}")

    # 최적 파라미터로 최종 학습
    print("\n[최종 학습 시작]")
    train_best(args.bird, features, best.params, args)
