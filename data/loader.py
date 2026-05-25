import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
from torch.utils.data import DataLoader, TensorDataset

from ai.common import (
    DATASET_PATH, FEAT_SCALER_PATH, TARGET_SCALER_PATH,
    TIMESTAMP_COL, ALL_FEATURES, TARGET_FEATURES, BIRDS
)

def _make_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """2D 배열을 슬라이딩 윈도우로 분할. (num_samples, F) → (num_windows, window_size, F)"""
    assert window_size <= len(data), \
        f"window_size({window_size})가 데이터 길이({len(data)})보다 클 수 없습니다."
    return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)])


def _extract_bird(data: pd.DataFrame, bird: str, features: list):
    """특정 새의 데이터를 시간순 정렬 후 feature/target 배열로 반환."""
    assert bird is not None, "bird는 빈 값이 아니여야 합니다."
    assert bird in data['bird'].unique(), f"bird는 {data['bird'].unique()} 중 하나여야 합니다."
    assert all(f in ALL_FEATURES for f in features), f"features는 {ALL_FEATURES} 중에서 선택되어야 합니다."

    subset = data[data['bird'] == bird].sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # NaN을 이전 값으로 채움 (시계열 연속성 유지)
    cols = features + TARGET_FEATURES
    nan_count = subset[cols].isna().sum().sum()
    if nan_count:
        subset[cols] = subset[cols].ffill().bfill()
        print(f"[경고] {bird}: NaN {nan_count}개를 ffill로 채웠습니다.")

    X = subset[features].values         # (num_samples, num_features)
    y = subset[TARGET_FEATURES].values  # (num_samples, num_targets)
    return X, y


def _collect_windows(data: pd.DataFrame, birds: list, features: list, window_size: int):
    """여러 개체의 윈도우를 개체 경계 없이 각각 생성 후 concat."""
    X_list, y_list = [], []
    for bird in birds:
        X, y = _extract_bird(data, bird, features)
        # 개체 내부에서만 윈도우 생성 → 개체 간 경계를 넘지 않음
        X_list.append(_make_windows(X, window_size))
        y_list.append(_make_windows(y, window_size))
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


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


def get_data_loader(features: list, window_size: int, batch_size: int = 32):
    """CSV 로드부터 DataLoader 반환까지의 전처리 파이프라인.

    분할 기준 (개체 단위, 시계열 오염 방지):
        Train : Art, Jill, Hudson, Bea, Caley, Isabel  (순풍형·광주기형 혼합)
        Val   : Whit                                    (순풍형 검증)
        Test  : Bergen                                  (광주기형 → 일반화 검증)
    """
    data = pd.read_csv(DATASET_PATH)

    # 개체별로 윈도우 생성 후 split별로 concat
    X_train, y_train = _collect_windows(data, BIRDS.get('train'), features, window_size)
    X_val,   y_val   = _collect_windows(data, BIRDS.get('valid'),   features, window_size)
    X_test,  y_test  = _collect_windows(data, BIRDS.get('test'),  features, window_size)

    X_train, y_train, X_val, y_val, X_test, y_test = _normalize(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    train_loader = _make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   batch_size=batch_size)
    test_loader  = _make_loader(X_test,  y_test,  batch_size=batch_size)

    return train_loader, val_loader, test_loader