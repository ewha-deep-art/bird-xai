import numpy as np
import torch
import joblib
from collections import deque
from itertools import cycle
from captum.attr import IntegratedGradients

from ai.common import (
    DEVICE, FEATURES, TARGET_FEATURES, FORECAST_HORIZON,
    FEAT_SCALER_PATH, DELTA_SCALER_PATH, test_loader, delta_to_absolute,
)
from ai.common.models import (
    FrameMessage,
    Point,
    AttributionFeatureKey,
    OverrideKey,
    XaiResult,
)
from ai.training.model import load_model_with_state

QUEUE_REFILL_THRESHOLD = 14
XAI_SOURCE_FEATURES = ("ws_850", "t_850", "q_850", "lapse_rate")
XAI_WEATHER_SOURCE = "lapse_rate"


class BirdPipeline:
    def __init__(self):
        self._loader_iter = cycle(test_loader)
        self._queue: deque[tuple[Point, XaiResult]] = deque()
        self._pending_queue: deque[tuple[Point, XaiResult]] | None = None
        self._last_overrides: dict[OverrideKey, int] | None = None
        self.model = load_model_with_state()
        self.feat_scaler = joblib.load(FEAT_SCALER_PATH)
        self.delta_scaler = joblib.load(DELTA_SCALER_PATH)
        self.ig = IntegratedGradients(self.model)

    def predict(self, X, last_obs: np.ndarray) -> list[Point]:
        with torch.no_grad():
            pred = self.model(X.to(DEVICE)).cpu().numpy()
        pred = pred.reshape(FORECAST_HORIZON, len(TARGET_FEATURES))
        deltas = self.delta_scaler.inverse_transform(pred)
        abs_pos = delta_to_absolute(last_obs, deltas)
        return [
            Point(lat=float(row[0]), lon=float(row[1]), altitude_m=float(row[2]))
            for row in abs_pos
        ]

    def _unity_attributions(self, row: np.ndarray, X) -> dict[AttributionFeatureKey, float]:
        abs_row = np.abs(row)
        total = float(abs_row.sum()) or 1.0
        source = {
            feature: float(abs_row[FEATURES.index(feature)] / total)
            for feature in XAI_SOURCE_FEATURES
        }

        X_inv = self.feat_scaler.inverse_transform(
            X.detach().cpu().numpy().reshape(-1, X.shape[-1])
        )
        ws_850_val = float(X_inv[-1, FEATURES.index("ws_850")])
        ws_attr = source["ws_850"]
        if ws_850_val >= 0:
            tailwind, headwind = ws_attr, 0.0
        else:
            tailwind, headwind = 0.0, ws_attr

        return {
            "tailwind": tailwind,
            "headwind": headwind,
            "weather_key": source[XAI_WEATHER_SOURCE],
        }

    def apply_xai(self, X) -> list[XaiResult]:
        X = X.to(DEVICE).requires_grad_(True)
        results = []

        for k in range(FORECAST_HORIZON):
            total_attrs = None
            for target_idx in range(len(TARGET_FEATURES)):
                attrs = self.ig.attribute(
                    X,
                    target=(k, target_idx),
                    n_steps=10,
                    return_convergence_delta=False,
                )
                total_attrs = attrs if total_attrs is None else total_attrs + attrs

            row = total_attrs[0].detach().cpu().numpy().mean(axis=0)
            results.append(XaiResult(attributions=self._unity_attributions(row, X)))
        return results

    def _apply_overrides_to_input(self, X, overrides: dict[OverrideKey, int]) -> torch.Tensor:
        cnt = max(10, min(100, overrides.get("message_cnt", 0)))
        ws_850 = (cnt - 10) * (34 / 90)
        X_inversed = self.feat_scaler.inverse_transform(X.cpu().numpy().reshape(-1, X.shape[-1]))
        X_inversed[:, FEATURES.index('ws_850')] = ws_850
        X_new = self.feat_scaler.transform(X_inversed).reshape(X.shape)
        return torch.tensor(X_new, dtype=X.dtype, device=X.device)

    def build_queue(self, overrides: dict[OverrideKey, int] | None) -> deque[tuple[Point, XaiResult]]:
        X, _, last_obs_batch = next(self._loader_iter)
        X = X[0:1]
        last_obs = last_obs_batch[0].numpy()

        if overrides:
            X = self._apply_overrides_to_input(X, overrides)

        future_points = self.predict(X, last_obs)
        xai_results = self.apply_xai(X)
        queue_items: list[tuple[Point, XaiResult]] = [
            (
                Point(
                    lat=float(last_obs[0]),
                    lon=float(last_obs[1]),
                    altitude_m=float(last_obs[2]),
                ),
                xai_results[0],
            )
        ]
        for k, point in enumerate(future_points):
            queue_items.append((point, xai_results[k]))
        return deque(queue_items)

    def build_frame_from_queue(self) -> FrameMessage | None:
        if self._pending_queue is not None:
            self._queue = self._pending_queue
            self._pending_queue = None

        if not self._queue:
            return None

        position, xai_result = self._queue.popleft()
        return FrameMessage(
            position=position,
            predicted_path=[p for p, _ in self._queue],
            candidates=[],
            xai=xai_result,
            applied_overrides=self._last_overrides,
        )

    def needs_prefill(self) -> bool:
        return len(self._queue) < QUEUE_REFILL_THRESHOLD and self._pending_queue is None
