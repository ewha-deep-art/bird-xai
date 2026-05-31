"""Terminal WebSocket smoke test for the Bird XAI server."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from pydantic import TypeAdapter

from ai.common.models import SCHEMA_VERSION, FrameMessage, ServerMessage

import websockets

_server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def parse_server_message(payload: dict[str, Any]) -> ServerMessage:
    return _server_message_adapter.validate_python(payload)


async def _receive_validated_message(socket, *, timeout: float) -> ServerMessage:
    raw = await asyncio.wait_for(socket.recv(), timeout=timeout)
    payload = json.loads(raw)
    message = parse_server_message(payload)
    print(f"<= {message.message_type}")
    return message


async def run_smoke_test(
    *,
    url: str,
    timeout: float,
) -> int:

    print(f"Connecting to {url}")
    async with websockets.connect(url, max_size=2**20) as socket:
        frame = _expect_frame(await _receive_validated_message(socket, timeout=timeout))
        print(
            "Bootstrap OK:"
            f" candidates={len(frame.candidates)}"
            f" xai_features={list(frame.xai.attributions.keys())}"
        )

        print("Streaming frames...")
        from datetime import datetime
        for i in range(180):
            frame = _expect_frame(await _receive_validated_message(socket, timeout=timeout))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] frame {i+1}: positions={frame.position}, xai={frame.xai}, applied_overrides={frame.applied_overrides}")
        
        frame = _expect_frame(await _receive_validated_message(socket, timeout=timeout))
        print(
            "Controls roundtrip OK:"
        )
        print("Smoke test passed.")
        return 0


def _expect_frame(message: ServerMessage) -> FrameMessage:
    if not isinstance(message, FrameMessage):
        raise RuntimeError(f"Expected frame, got {message.message_type}: {getattr(message, 'detail', '')}")
    return message


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a terminal WebSocket smoke test against Bird XAI.")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws", help="WebSocket endpoint URL.")
    parser.add_argument("--timeout", type=float, default=100.0, help="Seconds to wait for each server response.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(
        run_smoke_test(
            url=args.url,
            timeout=args.timeout,
        )
    )
    raise SystemExit(exit_code)
