"""Server orchestration helpers."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncGenerator

from ai.common.models import FrameMessage, parse_interaction_event
from ai.server.session import SessionState
from ai.inference.pipeline import BirdPipeline


class ServerService:
    def __init__(self) -> None:
        self.session = SessionState()
        self.startup_error: str | None = None
        self.pipeline: BirdPipeline | None = None
        self.lock = asyncio.Lock()
        try:
            self.pipeline = BirdPipeline()
        except Exception as exc:
            self.startup_error = str(exc)

    async def iter_frames(self) -> AsyncGenerator[FrameMessage, None]:
        loop = asyncio.get_running_loop()
        while True:
            async with self.lock:
                overrides = self.session.overrides
            frame = await loop.run_in_executor(
                None,
                functools.partial(self.pipeline.build_frame, overrides=overrides),
            )
            yield frame

    async def update_overrides(self, payload: dict) -> None:
        event = parse_interaction_event(payload)
        async with self.lock:
            self.session.overrides = event.overrides