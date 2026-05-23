# 경로
from pathlib import Path
import sys

HOME_DIR = Path(__file__).resolve().parents[2]

if str(HOME_DIR) not in sys.path:
    sys.path.insert(0, str(HOME_DIR))

DATA_DIR = HOME_DIR / "data" / "processed"
MODEL_DIR = HOME_DIR / "ai" / "training" / "model" / "weights"

DATASET_PATH = DATA_DIR / "preprocessed_9birds_full.csv"
FEAT_SCALER_PATH = DATA_DIR / "feat_scaler.pkl"
TARGET_SCALER_PATH = DATA_DIR / "target_scaler.pkl"

# 모델 관련 상수
import torch

RANDOM_STATE = 18
DEVICE = torch.device("cpu")

TIMESTAMP_COL = "timestamp"
ALL_FEATURES = [
    "lat", "lon", "height_raw",
    "ground_speed", "heading", "is_moving",
    "daylength_h", "u_925", "v_925",
    "q_850", "t_850", "t_925",
    "lapse_rate", "w_850",
]
TARGET_FEATURES = ["lat", "lon", "height_raw"]

BIRD = "Art" # NOTE: 변경 가능
FEATURES = ["daylength_h", "u_925", "v_925"] # NOTE: 변경 가능
WINDOW_SIZE = 24 # NOTE: 변경 가능

# 데이터 로더
from data.loader import get_data_loader

train_loader, val_loader, test_loader = get_data_loader(
    bird=BIRD,
    features=FEATURES,
    window_size=WINDOW_SIZE,
)