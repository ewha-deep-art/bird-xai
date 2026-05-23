from typing import get_args
import torch
import joblib
from collections import deque
from itertools import cycle
from captum.attr import IntegratedGradients

from ai.common import DEVICE, FEATURES, TARGET_FEATURES, TARGET_SCALER_PATH, BIRD, test_loader
from ai.common.models import (
    CandidatePath,
    FrameMessage,
    Point,
    AttributionFeatureKey,
    XaiResult,
)
from ai.training.model import load_model_with_state

QUEUE_REFILL_THRESHOLD = 10


class BirdPipeline:
    backend_name = "bird"

    def __init__(self):
        self._loader_iter = cycle(test_loader)  # NOTE: test_loader가 크다면 메모리 부담 있음
        self._queue: deque[tuple[Point, XaiResult]] = deque()
        self._pending_queue: deque[tuple[Point, XaiResult]] | None = None
        self._last_overrides: dict | None = None
        self.model = load_model_with_state(bird=BIRD)
        self.target_scaler = joblib.load(TARGET_SCALER_PATH)
        self.ig = IntegratedGradients(self.model)

    def predict(self, X) -> list[Point]:
        with torch.no_grad():
            pred = self.model(X.to(DEVICE)).cpu().numpy()
        pred = pred.reshape(-1, len(TARGET_FEATURES))
        pred = self.target_scaler.inverse_transform(pred)
        return [Point(lat=row[0], lon=row[1], altitude_m=row[2]) for row in pred]

    def apply_xai(self, X) -> list[XaiResult]:
        X = X.to(DEVICE).requires_grad_(True)

        # timestep × target 조합별로 attribution 계산 후 합산
        total_attrs = None
        n_timesteps = X.shape[1]  # 24
        for t in range(n_timesteps):
            for target_idx in range(len(TARGET_FEATURES)):  # 3
                attrs = self.ig.attribute(
                    X,
                    target=(t, target_idx),
                    n_steps=50,
                    return_convergence_delta=False,
                )
                total_attrs = attrs if total_attrs is None else total_attrs + attrs

        # (batch=32, timestep=24, n_input_features)
        attrs_np = total_attrs.detach().cpu().numpy()
        attrs_np = attrs_np.reshape(-1, attrs_np.shape[-1])  # (batch*timestep, n_input_features)

        results = []
        for row in attrs_np:
            total = abs(row).sum() or 1.0
            normalized = {
                feature: float(abs(row[FEATURES.index(feature)]) / total)
                for feature in get_args(AttributionFeatureKey)
            }
            results.append(XaiResult(attributions=normalized))
        return results
    
    def _apply_overrides_to_input(self, X, overrides: dict) -> ...:
        # TODO: overrides 값으로 X의 wind_speed, wind_direction 피처 수정
        return X

    def _build_queue(self, overrides: dict | None) -> deque[tuple[Point, XaiResult]]:
        X, _ = next(self._loader_iter)
        if overrides:
            X = self._apply_overrides_to_input(X, overrides)
        points = self.predict(X)
        xai_results = self.apply_xai(X)
        return deque(zip(points, xai_results))

    def build_frame(self, *, overrides: dict | None = None) -> FrameMessage:
        if overrides != self._last_overrides and self._pending_queue is None:
            self._pending_queue = self._build_queue(overrides)
            self._last_overrides = overrides

        if self._pending_queue is not None:
            self._queue = self._pending_queue
            self._pending_queue = None

        if len(self._queue) < QUEUE_REFILL_THRESHOLD:
            self._queue.extend(self._build_queue(overrides))

        position, xai_result = self._queue.popleft()

        return FrameMessage(
            position=position,
            predicted_path=[p for p, _ in self._queue],  # position 이후의 경로
            candidates=[], # TODO: 모델 구조 바꿔서 채우기
            xai=xai_result,
            applied_overrides=overrides or None,
        )