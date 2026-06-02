"""FastAPI application for Bird XAI."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ai.config import get_settings
from ai.common.models import ErrorMessage


def create_app() -> FastAPI:
    app = FastAPI(title="Bird XAI Server", version="0.1.0")
    settings = get_settings()
    from ai.server.service import ServerService
    service = ServerService()

    @app.get("/health")
    async def health() -> dict:
        if service.startup_error is not None or service.pipeline is None:
            return {
                "status": "error",
                "detail": service.startup_error,
            }
        return {
            "status": "ok",
            "backend": service.pipeline.backend_name,
            "subject_id": service.session.subject_id,
        }

    @app.post("/wish")
    async def wish(message: str) -> dict:
        if service.startup_error is not None or service.pipeline is None:
            return {
                "status": "error",
                "detail": service.startup_error,
            }
        if message:
            await service.update_overrides()
            return {
                "status": "ok",
                "backend": service.pipeline.backend_name,
                "subject_id": service.session.subject_id,
            }
        return {
            "status": "error",
            "detail": "message is empty or not a string",
        }
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()

        if service.startup_error is not None or service.pipeline is None:
            err = ErrorMessage(code="startup_error", detail=service.startup_error or "pipeline unavailable")
            await websocket.send_text(err.model_dump_json())
            return

        async def send_loop() -> None:
            async for frame in service.iter_frames():
                await websocket.send_text(frame.model_dump_json())
                await asyncio.sleep(settings.frame_interval)

        send_task = asyncio.create_task(send_loop())
        try:
            while True:
                await websocket.receive_json() # 클라이언트로부터의 메시지는 현재 사용하지 않지만, 연결 유지를 위해 수신 대기
        except WebSocketDisconnect:
            pass
        finally:
            send_task.cancel()
            with suppress(asyncio.CancelledError):
                await send_task

    return app

def main() -> None:
    settings = get_settings()
    import uvicorn

    uvicorn.run(
        "ai.server.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=False,
    )
