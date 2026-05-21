"""Server orchestration helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from pathlib import Path

from ai.contracts.models import (
    FrameMessage,
    parse_interaction_event,
)
from ai.server.session import SessionState

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "contracts" / "examples" / "frame.sample.json"
)


class ServerService:
    def __init__(self) -> None:
        self.session = SessionState()
        self.startup_error: str | None = None

        if os.getenv("BIRD_XAI_MOCK", "").lower() in ("true", "1"):
            self.pipeline = _MockPipeline()
        else:
            try:
                from ai.inference.pipeline import build_pipeline
                self.pipeline = build_pipeline()
            except Exception as exc:
                self.startup_error = str(exc)
                self.pipeline = None

    def iter_frames(self) -> Generator[FrameMessage, None, None]:
        while True:
            yield self.pipeline.build_frame(overrides=self.session.overrides)

    def update_overrides(self, payload: dict) -> None:
        event = parse_interaction_event(payload)
        self.session.overrides = event.overrides


class _MockPipeline:
    backend_name = "mock"

    def __init__(self) -> None:
        raw = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
        self._base_frame = FrameMessage.model_validate(raw)

    def build_frame(self, *, overrides: dict | None) -> FrameMessage:
        return self._base_frame.model_copy(update={"applied_overrides": overrides or None})
