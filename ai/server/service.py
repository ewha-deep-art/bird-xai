"""Server orchestration helpers."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncGenerator

from ai.common.models import FrameMessage, OverrideKey
from ai.config import get_settings
from ai.inference.pipeline import BirdPipeline


class ServerService:
    subject_id: str = "White-fronted Goose"

    def __init__(self) -> None:
        self.startup_error: str | None = None
        self.pipeline: BirdPipeline | None = None
        self.message_counter: int = 0
        self._flush_threshold = get_settings().wish_flush_threshold
        try:
            self.pipeline = BirdPipeline()
        except Exception as exc:
            self.startup_error = str(exc)

    def parse_overrides(self) -> dict[OverrideKey, int] | None:
        if self.message_counter >= self._flush_threshold:
            overrides = {"message_cnt": self.message_counter}
            self.message_counter = 0
            return overrides
        return None

    async def iter_frames(self) -> AsyncGenerator[FrameMessage, None]:
        loop = asyncio.get_running_loop()
        pipeline = self.pipeline
        pending_build_task: asyncio.Task | None = None
        active_overrides: dict[OverrideKey, int] | None = self.parse_overrides()

        async def run_build(overrides: dict[OverrideKey, int] | None):
            return await loop.run_in_executor(
                None,
                functools.partial(pipeline.build_queue, overrides),
            )

        pipeline._queue = await run_build(active_overrides)
        if active_overrides is not None:
            pipeline._last_overrides = active_overrides

        while True:
            new_overrides = self.parse_overrides()
            if new_overrides is not None:
                active_overrides = new_overrides

            can_schedule = pending_build_task is None or pending_build_task.done()

            if active_overrides != pipeline._last_overrides and can_schedule:
                ov = active_overrides

                async def swap_pending():
                    pipeline._pending_queue = await run_build(ov)
                    pipeline._last_overrides = ov

                pending_build_task = asyncio.create_task(swap_pending())

            elif pipeline.needs_prefill() and can_schedule:
                ov = active_overrides

                async def extend_queue():
                    pipeline._queue.extend(await run_build(ov))

                pending_build_task = asyncio.create_task(extend_queue())

            frame = pipeline.build_frame_from_queue()
            if frame is None:
                if pending_build_task is not None and not pending_build_task.done():
                    await pending_build_task
                else:
                    await asyncio.sleep(0.01)
                continue
            yield frame

    def update_overrides(self) -> None:
        self.message_counter += 1
