import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from torch.utils.data import DataLoader, TensorDataset

from ai.common import (
    RANDOM_STATE, TIMESTAMP_COL, ALL_FEATURES, TARGET_FEATURES,
    DATASET_PATH, FEAT_SCALER_PATH, TARGET_SCALER_PATH,
)


def _make_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """2D 배열을 슬라이딩 윈도우로 분할. (num_samples, F) → (num_windows, window_size, F)"""
    assert window_size <= len(data), \
        f"window_size({window_size})가 데이터 길이({len(data)})보다 클 수 없습니다."
    return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)])


def _select(data: pd.DataFrame, bird: str, features: list):
    """특정 새의 데이터를 시간순 정렬 후 feature/target 배열로 반환."""
    assert bird in data['bird'].unique(), f"bird는 {data['bird'].unique()} 중 하나여야 합니다."
    assert all(f in ALL_FEATURES for f in features), f"features는 {ALL_FEATURES} 중에서 선택되어야 합니다."

    data = data[data['bird'] == bird]
    data = data.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # NaN을 이전 값으로 채움 (시계열 연속성 유지)
    cols = features + TARGET_FEATURES
    nan_count = data[cols].isna().sum().sum()
    if nan_count:
        data[cols] = data[cols].ffill().bfill()
        print(f"[경고] NaN {nan_count}개를 ffill로 채웠습니다.")

    X = data[features].values         # (num_samples, num_features)
    y = data[TARGET_FEATURES].values  # (num_samples, num_targets)
    return X, y


def _split_train_eval(X, y):
    """시간 순서를 유지하며 train(80%) / val(10%) / test(10%) 분할."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=False)
    X_val, X_test, y_val, y_test   = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE, shuffle=False)
    return X_train, y_train, X_val, y_val, X_test, y_test


def _normalize(X_train, y_train, X_val, y_val, X_test, y_test):
    """train 기준으로 MinMaxScaler fit 후 전체 분할에 적용. scaler는 파일로 저장."""
    feat_scaler   = MinMaxScaler()
    target_scaler = MinMaxScaler()

    def transform_X(X, fit=False):
        n, ws, nf = X.shape
        d = X.reshape(-1, nf)
        return (feat_scaler.fit_transform(d) if fit else feat_scaler.transform(d)).reshape(n, ws, nf)

    def transform_y(y, fit=False):
        n, ws, nt = y.shape
        d = y.reshape(-1, nt)
        return (target_scaler.fit_transform(d) if fit else target_scaler.transform(d)).reshape(n, ws, nt)

    X_train = transform_X(X_train, fit=True)
    X_val   = transform_X(X_val)
    X_test  = transform_X(X_test)

    y_train = transform_y(y_train, fit=True)
    y_val   = transform_y(y_val)
    y_test  = transform_y(y_test)

    joblib.dump(feat_scaler,   FEAT_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)

    return X_train, y_train, X_val, y_val, X_test, y_test


def _make_loader(X, y, batch_size=32, shuffle=False):
    """numpy 배열을 PyTorch DataLoader로 변환."""
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)


def get_data_loader(bird: str, features: list, window_size: int, batch_size: int = 32):
    """CSV 로드부터 DataLoader 반환까지의 전처리 파이프라인."""
    data = pd.read_csv(DATASET_PATH)

    X, y = _select(data, bird=bird, features=features)

    # 슬라이딩 윈도우 적용 → (num_windows, window_size, num_features/num_targets)
    X = _make_windows(X, window_size)
    y = _make_windows(y, window_size)

    X_train, y_train, X_val, y_val, X_test, y_test = _split_train_eval(X, y)
    X_train, y_train, X_val, y_val, X_test, y_test = _normalize(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    train_loader = _make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   batch_size=batch_size)
    test_loader  = _make_loader(X_test,  y_test,  batch_size=batch_size)

    return train_loader, val_loader, test_loader