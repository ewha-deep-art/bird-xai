import torch
import torch.nn as nn

from ai.common import DEVICE, MODEL_SAVE_PATH, FEATURES, TARGET_FEATURES, FORECAST_HORIZON


class BirdForecastLSTM(nn.Module):
    """LSTM encoder + direct multi-step future head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        forecast_horizon: int,
        output_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, forecast_horizon * output_size)

    def forward(self, x):
        # x: (batch, window_size, input_size)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        out = self.fc(h)
        return out.view(x.size(0), self.forecast_horizon, self.output_size)


def load_model():
    model = BirdForecastLSTM(
        input_size=len(FEATURES),
        hidden_size=256,
        num_layers=3,
        forecast_horizon=FORECAST_HORIZON,
        output_size=len(TARGET_FEATURES),
        dropout=0.3,
    ).to(DEVICE)
    return model


def load_model_with_state():
    model = load_model()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    return model
