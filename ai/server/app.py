"""FastAPI application for Bird XAI."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from ai.config import get_settings
from ai.common.models import ErrorMessage, WishRequest, WishResponse
from ai.server.rate_limit import RateLimiter

VISITOR_HTML = Path(__file__).parent / "static" / "participate.html"


def create_app() -> FastAPI:
    app = FastAPI(title="Bird XAI Server", version="0.1.0")
    settings = get_settings()
    from ai.server.service import ServerService
    service = ServerService()
    rate_limiter = RateLimiter(interval_sec=settings.wish_rate_limit_sec)

    @app.exception_handler(RequestValidationError)
    async def validation_to_bad_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.errors()})

    @app.get("/wind-and-wish", include_in_schema=False)
    async def wind_and_wish_page() -> FileResponse:
        return FileResponse(VISITOR_HTML, media_type="text/html; charset=utf-8")

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
            "subject_id": service.subject_id,
        }

    @app.post("/wish", dependencies=[Depends(rate_limiter)])
    async def wish(_body: WishRequest) -> WishResponse:
        if service.startup_error is not None or service.pipeline is None:
            raise HTTPException(status_code=503, detail=service.startup_error)

        service.update_overrides()
        return WishResponse(
            status="ok",
            backend=service.pipeline.backend_name,
            subject_id=service.subject_id,
        )

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
                await websocket.receive_json()
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
