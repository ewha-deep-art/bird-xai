import torch
import torch.nn as nn

from ai.common import DEVICE, MODEL_DIR, FEATURES, TARGET_FEATURES

class BirdLSTM(nn.Module):
    """단층/다층 LSTM + FC 출력."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, window_size, input_size)
        out, _ = self.lstm(x)
        return self.fc(out)  # (batch, window_size, output_size)
    
def load_model():
    model = BirdLSTM(
        input_size=len(FEATURES),
        hidden_size=64,
        num_layers=3,
        output_size=len(TARGET_FEATURES),
        dropout=0.3,
    ).to(DEVICE)
    return model

def load_model_with_state(bird: str):
    model = load_model()
    model.load_state_dict(torch.load(MODEL_DIR / f"{bird}_best.pt", map_location=DEVICE))
    model.eval()
    return model