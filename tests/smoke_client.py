"""Terminal WebSocket smoke test for the Bird XAI server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from pydantic import TypeAdapter
import websockets

from ai.common.models import SCHEMA_VERSION, FrameMessage, ServerMessage

_server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def parse_server_message(payload: dict[str, Any]) -> ServerMessage:
    return _server_message_adapter.validate_python(payload)


def ws_url_from_base(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, "/ws", "", "", ""))


async def _receive_validated_message(socket, *, timeout: float) -> ServerMessage:
    raw = await asyncio.wait_for(socket.recv(), timeout=timeout)
    payload = json.loads(raw)
    message = parse_server_message(payload)
    print(f"<= {message.message_type}")
    return message


async def _post_wish(client: httpx.AsyncClient, *, message: str) -> None:
    response = await client.post("/wish", json={"message": message})
    response.raise_for_status()
    print(f"POST /wish -> {response.status_code}")


async def _run_wish_override_scenario(
    *,
    base_url: str,
    ws_url: str,
    timeout: float,
    flush_threshold: int,
    wish_interval: float,
) -> None:
    print(f"Wish override scenario (threshold={flush_threshold})")
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        async with websockets.connect(ws_url, max_size=2**20) as socket:
            frame = _expect_frame(await _receive_validated_message(socket, timeout=timeout))
            if frame.applied_overrides is not None:
                raise RuntimeError(f"Expected no overrides at bootstrap, got {frame.applied_overrides}")

            for i in range(flush_threshold):
                await _post_wish(client, message="바람")
                if i + 1 < flush_threshold and wish_interval > 0:
                    await asyncio.sleep(wish_interval)

            print("Waiting for applied_overrides on frame stream...")
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                frame = _expect_frame(await _receive_validated_message(socket, timeout=timeout))
                if frame.applied_overrides is not None:
                    message_cnt = frame.applied_overrides.get("message_cnt")
                    if message_cnt is None or message_cnt < flush_threshold:
                        raise RuntimeError(
                            f"Expected message_cnt>={flush_threshold}, got {frame.applied_overrides}"
                        )
                    print(f"Override OK: applied_overrides={frame.applied_overrides}")
                    return
            raise RuntimeError(f"Timed out waiting for applied_overrides after {timeout}s")


async def _check_visitor_page(*, base_url: str, timeout: float) -> None:
    print("GET /wind-and-wish")
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        response = await client.get("/wind-and-wish")
        if response.status_code != 200:
            raise RuntimeError(f"Expected 200 for /wind-and-wish, got {response.status_code}")
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            raise RuntimeError(f"Expected text/html for /wind-and-wish, got {content_type!r}")
        body = response.text
        for snippet in ("바람 보내기", "철새에게 전하세요"):
            if snippet not in body:
                raise RuntimeError(f"Expected {snippet!r} in /wind-and-wish body")
    print("Visitor page OK")


async def _run_negative_wish_checks(*, base_url: str, timeout: float) -> None:
    print("Negative /wish checks")
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        empty = await client.post("/wish", json={})
        if empty.status_code != 400:
            raise RuntimeError(f"Expected 400 for empty body, got {empty.status_code}")

        await _post_wish(client, message="ok")
        rapid = await client.post("/wish", json={"message": "again"})
        if rapid.status_code != 429:
            raise RuntimeError(f"Expected 429 for rapid repeat, got {rapid.status_code}")
    print("Negative checks OK")


async def run_smoke_test(
    *,
    base_url: str,
    url: str | None,
    timeout: float,
    frame_count: int,
    with_wish: bool,
    with_negative: bool,
    flush_threshold: int,
    wish_interval: float,
) -> int:
    ws_url = url or ws_url_from_base(base_url)
    print(f"Connecting to {ws_url}")

    await _check_visitor_page(base_url=base_url, timeout=timeout)

    if with_negative:
        await _run_negative_wish_checks(base_url=base_url, timeout=timeout)

    if with_wish:
        await _run_wish_override_scenario(
            base_url=base_url,
            ws_url=ws_url,
            timeout=timeout,
            flush_threshold=flush_threshold,
            wish_interval=wish_interval,
        )

    async with websockets.connect(ws_url, max_size=2**20) as socket:
        frame = _expect_frame(await _receive_validated_message(socket, timeout=timeout))
        expected_xai = {"tailwind", "headwind", "weather_key"}
        if set(frame.xai.attributions.keys()) != expected_xai:
            raise RuntimeError(f"Expected xai keys {expected_xai}, got {set(frame.xai.attributions)}")
        print(
            "Bootstrap OK:"
            f" schema={SCHEMA_VERSION}"
            f" candidates={len(frame.candidates)}"
            f" xai_features={list(frame.xai.attributions.keys())}"
        )

        if frame_count > 0:
            print(f"Streaming {frame_count} frames...")
            for i in range(frame_count):
                frame = _expect_frame(await _receive_validated_message(socket, timeout=timeout))
                print(
                    f"frame {i + 1}:"
                    f" position=({frame.position.lat:.4f},{frame.position.lon:.4f})"
                    f" applied_overrides={frame.applied_overrides}"
                )

            frame = _expect_frame(await _receive_validated_message(socket, timeout=timeout))
            print("Frame stream OK.")

    print("Smoke test passed.")
    return 0


def _expect_frame(message: ServerMessage) -> FrameMessage:
    if not isinstance(message, FrameMessage):
        raise RuntimeError(
            f"Expected frame, got {message.message_type}: {getattr(message, 'detail', '')}"
        )
    return message


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a terminal smoke test against Bird XAI.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="HTTP base URL for /wish and /health.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="WebSocket endpoint URL (default: derived from --base-url).",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="Seconds to wait per step.")
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of frames to stream after wish scenario (0 to skip).",
    )
    parser.add_argument(
        "--with-wish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run /wish flush -> applied_overrides scenario.",
    )
    parser.add_argument(
        "--with-negative",
        action="store_true",
        help="Run 400/429 negative /wish checks (uses rate limit; run before wish scenario).",
    )
    parser.add_argument(
        "--flush-threshold",
        type=int,
        default=10,
        help="Expected message_cnt flush threshold (match BIRD_XAI_WISH_FLUSH_THRESHOLD).",
    )
    parser.add_argument(
        "--wish-interval",
        type=float,
        default=0.0,
        help="Seconds between /wish posts (use 5+ for production rate limit; 0 for CI).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(
            run_smoke_test(
                base_url=args.base_url.rstrip("/"),
                url=args.url,
                timeout=args.timeout,
                frame_count=args.frames,
                with_wish=args.with_wish,
                with_negative=args.with_negative,
                flush_threshold=args.flush_threshold,
                wish_interval=args.wish_interval,
            )
        )
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
