import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
from torch.utils.data import DataLoader, TensorDataset

from ai.common import (
    DATASET_PATH, FEAT_SCALER_PATH, DELTA_SCALER_PATH,
    TIMESTAMP_COL, ALL_FEATURES, TARGET_FEATURES
)

HEIGHT_CLIP_MIN = 0.0
HEIGHT_CLIP_MAX = 2000.0


def absolute_to_delta(last_obs: np.ndarray, future_abs: np.ndarray) -> np.ndarray:
    """절대 좌표 시퀀스를 step별 Δ로 변환. last_obs (3,), future_abs (M, 3) → (M, 3)."""
    deltas = np.zeros_like(future_abs)
    prev = last_obs.copy()
    for k in range(len(future_abs)):
        deltas[k] = future_abs[k] - prev
        prev = future_abs[k]
    return deltas


def delta_to_absolute(last_obs: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Δ 시퀀스를 절대 좌표로 누적. last_obs (3,), deltas (M, 3) → (M, 3)."""
    positions = np.zeros_like(deltas)
    current = last_obs.copy()
    for k in range(len(deltas)):
        current = current + deltas[k]
        positions[k] = current
    return positions


def _make_forecast_pairs(
    X: np.ndarray, y: np.ndarray, input_len: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """과거 input_len step → 미래 horizon step Δ target 쌍 생성."""
    pairs_X, pairs_y, pairs_last = [], [], []
    for i in range(len(X) - input_len - horizon + 1):
        pairs_X.append(X[i:i + input_len])
        future_abs = y[i + input_len:i + input_len + horizon]
        last_obs = y[i + input_len - 1]
        pairs_y.append(absolute_to_delta(last_obs, future_abs))
        pairs_last.append(last_obs)
    if not pairs_X:
        raise ValueError(
            f"forecast pair 없음: len={len(X)}, input_len={input_len}, horizon={horizon}"
        )
    return np.array(pairs_X), np.array(pairs_y), np.array(pairs_last)


def _extract_bird(data: pd.DataFrame, bird: str, features: list):
    """특정 새의 데이터를 시간순 정렬 후 feature/target 배열로 반환."""
    assert bird is not None, "bird는 빈 값이 아니여야 합니다."
    assert bird in data['bird'].unique(), f"bird는 {data['bird'].unique()} 중 하나여야 합니다."
    assert all(f in ALL_FEATURES for f in features), f"features는 {ALL_FEATURES} 중에서 선택되어야 합니다."

    subset = data[data['bird'] == bird].sort_values(TIMESTAMP_COL).reset_index(drop=True)

    cols = list(dict.fromkeys(features + TARGET_FEATURES))
    nan_count = subset[cols].isna().sum().sum()
    if nan_count:
        subset[cols] = subset[cols].ffill().bfill().fillna(0)
        print(f"[경고] {bird}: NaN {nan_count}개를 ffill로 채웠습니다.")

    subset["height_raw"] = subset["height_raw"].clip(HEIGHT_CLIP_MIN, HEIGHT_CLIP_MAX)

    X = subset[features].values
    y = subset[TARGET_FEATURES].values
    return X, y


def _collect_forecast_pairs(
    data: pd.DataFrame, birds: list, features: list, input_len: int, horizon: int
):
    """여러 개체의 forecast pair를 생성 후 concat."""
    X_list, y_list, last_list = [], [], []
    for bird in birds:
        X, y = _extract_bird(data, bird, features)
        px, py, pl = _make_forecast_pairs(X, y, input_len, horizon)
        X_list.append(px)
        y_list.append(py)
        last_list.append(pl)
    return (
        np.concatenate(X_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(last_list, axis=0),
    )


def _normalize(X_train, y_train, X_val, y_val, X_test, y_test):
    """train 기준으로 MinMaxScaler fit 후 전체 분할에 적용. scaler는 파일로 저장."""
    feat_scaler = MinMaxScaler()
    delta_scaler = MinMaxScaler()

    def transform_X(X, fit=False):
        n, ws, nf = X.shape
        d = X.reshape(-1, nf)
        return (feat_scaler.fit_transform(d) if fit else feat_scaler.transform(d)).reshape(n, ws, nf)

    def transform_y(y, fit=False):
        n, horizon, nt = y.shape
        d = y.reshape(-1, nt)
        return (delta_scaler.fit_transform(d) if fit else delta_scaler.transform(d)).reshape(n, horizon, nt)

    X_train = transform_X(X_train, fit=True)
    X_val = transform_X(X_val)
    X_test = transform_X(X_test)

    y_train = transform_y(y_train, fit=True)
    y_val = transform_y(y_val)
    y_test = transform_y(y_test)

    joblib.dump(feat_scaler, FEAT_SCALER_PATH)
    joblib.dump(delta_scaler, DELTA_SCALER_PATH)

    return X_train, y_train, X_val, y_val, X_test, y_test


def _make_loader(X, y, last_obs, batch_size=32, shuffle=False):
    """numpy 배열을 PyTorch DataLoader로 변환. last_obs는 절대 좌표 (batch, 3)."""
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    t_last = torch.tensor(last_obs, dtype=torch.float32)
    return DataLoader(TensorDataset(tx, ty, t_last), batch_size=batch_size, shuffle=shuffle)


def get_data_loader(
    features: list,
    window_size: int,
    forecast_horizon: int,
    batch_size: int = 32,
):
    """CSV 로드부터 DataLoader 반환까지의 forecast 전처리 파이프라인."""
    data = pd.read_csv(DATASET_PATH)

    birds_to_exclude = [
        '701', '707', '709', '711', '712', '720', '738', '742', '749', '750',
        '766', '768', 'CS007027_3098', 'Frank_3084', 'Hannah_3988_LK2',
    ]
    birds_in_data = sorted([
        bird for bird in data['bird'].unique() if bird not in birds_to_exclude
    ])
    birds_in_data_len = len(birds_in_data)
    birds = {
        "train": birds_in_data[:int(birds_in_data_len * 0.7)],
        "valid": birds_in_data[int(birds_in_data_len * 0.7):int(birds_in_data_len * 0.85)],
        "test": birds_in_data[int(birds_in_data_len * 0.85):],
    }

    X_train, y_train, last_train = _collect_forecast_pairs(
        data, birds.get('train'), features, window_size, forecast_horizon
    )
    X_val, y_val, last_val = _collect_forecast_pairs(
        data, birds.get('valid'), features, window_size, forecast_horizon
    )
    X_test, y_test, last_test = _collect_forecast_pairs(
        data, birds.get('test'), features, window_size, forecast_horizon
    )

    X_train, y_train, X_val, y_val, X_test, y_test = _normalize(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    train_loader = _make_loader(X_train, y_train, last_train, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, last_val, batch_size=batch_size)
    test_loader = _make_loader(X_test, y_test, last_test, batch_size=batch_size)

    return train_loader, val_loader, test_loader
