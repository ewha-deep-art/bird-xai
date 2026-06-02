# 경로
from pathlib import Path
import sys

HOME_DIR = Path(__file__).resolve().parents[2]

if str(HOME_DIR) not in sys.path:
    sys.path.insert(0, str(HOME_DIR))

DATA_DIR = HOME_DIR / "data" / "processed"

MODEL_SAVE_PATH = HOME_DIR / "ai" / "training" / "model" / "weights" / "bird_best.pt"
DATASET_PATH = DATA_DIR / "preprocessed_geese_full.csv"
FEAT_SCALER_PATH = DATA_DIR / "feat_scaler.pkl"
DELTA_SCALER_PATH = DATA_DIR / "delta_scaler.pkl"
# legacy alias — delta_scaler.pkl과 동일 파일
TARGET_SCALER_PATH = DELTA_SCALER_PATH

# 모델 관련 상수
import torch

RANDOM_STATE = 18
DEVICE = torch.device("cpu")

TIMESTAMP_COL = "timestamp"
ALL_FEATURES = [
    'bird', 'timestamp', 'is_interpolated_gps', 'is_interpolated_era5',
    'lat', 'lon', 'height_raw', 'ground_speed', 'heading', 'is_moving',
    'u_1000', 'v_1000', 'w_1000', 't_1000', 'z_1000', 'q_1000', 'r_1000',
    'cc_1000', 'u_925', 'v_925', 'w_925', 't_925', 'z_925', 'q_925',
    'r_925', 'cc_925', 'u_850', 'v_850', 'w_850', 't_850', 'z_850', 'q_850',
    'r_850', 'cc_850', 'ws_1000', 'ws_925', 'ws_850', 'wspeed_1000',
    'wspeed_925', 'wspeed_850', 'wdir_1000', 'wdir_925', 'wdir_850', 'lapse_rate'
]
TARGET_FEATURES = ["lat", "lon", "height_raw"]

FEATURES = [
    'lat', 'lon', 'ground_speed', 'heading',
    'ws_850', 't_850', 'q_850', 'lapse_rate',
]
WINDOW_SIZE = 24 # NOTE: 변경 가능
FORECAST_HORIZON = 12

# 데이터 로더
from data.loader import get_data_loader, delta_to_absolute

train_loader, val_loader, test_loader = get_data_loader(
    features=FEATURES,
    window_size=WINDOW_SIZE,
    forecast_horizon=FORECAST_HORIZON,
)