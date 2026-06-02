import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import optuna
from optuna.trial import Trial

import time
import argparse
from pathlib import Path

from ai.common import DEVICE, FEATURES, TARGET_FEATURES, FORECAST_HORIZON, train_loader, val_loader, test_loader
from ai.training.model import BirdForecastLSTM
from ai.training.train import train_one_epoch, evaluate, test

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def make_objective(features: list, args):
    def objective(trial: Trial) -> float:
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        model = BirdForecastLSTM(
            input_size=len(features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            forecast_horizon=FORECAST_HORIZON,
            output_size=len(TARGET_FEATURES),
            dropout=dropout,
        ).to(DEVICE)

        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        best_val_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, DEVICE)
            val_loss = evaluate(model, val_loader, DEVICE)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_loss

    return objective


def train_best(features: list, best_params: dict, args):
    model = BirdForecastLSTM(
        input_size=len(features),
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        forecast_horizon=FORECAST_HORIZON,
        output_size=len(TARGET_FEATURES),
        dropout=best_params["dropout"],
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=best_params["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    checkpoint_path = CHECKPOINT_DIR / "bird_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_loss)

        print(f"[{epoch:03d}/{args.epochs}] train={train_loss:.4f}  val={val_loss:.4f}  ({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → checkpoint saved (val={best_val_loss:.4f})")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test(model, test_loader)


def parse_args():
    parser = argparse.ArgumentParser(description="Bird Forecast LSTM + Optuna Experiment")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n_trials", type=int, default=30, help="탐색할 trial 수")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    features = FEATURES

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", pruner=pruner)

    print(f"[Optuna] n_trials={args.n_trials}")
    study.optimize(
        make_objective(features, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n[탐색 완료] best trial #{best.number}  val_loss={best.value:.4f}")
    print(f"  파라미터: {best.params}")

    print("\n[최종 학습 시작]")
    train_best(features, best.params, args)
