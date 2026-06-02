"""Server orchestration helpers."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncGenerator

from ai.common.models import FrameMessage, OverrideKey
from ai.server.session import SessionState
from ai.inference.pipeline import BirdPipeline


class ServerService:
    def __init__(self) -> None:
        self.session = SessionState()
        self.startup_error: str | None = None
        self.pipeline: BirdPipeline | None = None
        self.message_counter: int = 0
        self.lock = asyncio.Lock()
        try:
            self.pipeline = BirdPipeline()
        except Exception as exc:
            self.startup_error = str(exc)

    async def parse_overrides(self) -> dict[OverrideKey, float] | None:
        if self.message_counter > 10:
            overrides = {
                "message_cnt": self.message_counter,
            }
            self.message_counter = 0
            return overrides
        return None
    
    async def iter_frames(self) -> AsyncGenerator[FrameMessage, None]:
        loop = asyncio.get_running_loop()
        pending_build_task: asyncio.Task | None = None
        current_overrides: dict | None = None

        # 초기 큐 채우기
        initial_queue = await loop.run_in_executor(
            None,
            functools.partial(self.pipeline.build_queue_blocking, None),
        )
        self.pipeline._queue = initial_queue

        while True:
            async with self.lock:
                new_overrides = await self.parse_overrides()
            
            if new_overrides is not None:
                current_overrides = new_overrides

            # override 변경 감지 → 백그라운드 빌드 시작 (한 번만)
            if (
                current_overrides != self.pipeline._last_overrides
                and (pending_build_task is None or pending_build_task.done())
            ):
                overrides_snapshot = current_overrides
                async def _build_and_set(ov=overrides_snapshot):
                    q = await loop.run_in_executor(
                        None,
                        functools.partial(self.pipeline.build_queue_blocking, ov),
                    )
                    self.pipeline.set_pending_queue(q, ov)

                pending_build_task = asyncio.create_task(_build_and_set())

            # 큐 부족 시 refill (override 없을 때)
            if (
                self.pipeline.needs_prefill()
                and (pending_build_task is None or pending_build_task.done())
            ):
                async def _refill(ov=current_overrides):
                    q = await loop.run_in_executor(
                        None,
                        functools.partial(self.pipeline.build_queue_blocking, ov),
                    )
                    self.pipeline._queue.extend(q)

                pending_build_task = asyncio.create_task(_refill())

            # 블로킹 없이 바로 꺼내서 yield
            frame, swapped = self.pipeline.build_frame_from_queue()
            if swapped:
                current_overrides = None
            yield frame

    async def update_overrides(self) -> None:
        self.message_counter += 1