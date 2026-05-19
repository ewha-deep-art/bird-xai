"""Pydantic models for all Bird XAI contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ai.common.constants import SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

AttributionFeatureKey = Literal["wind_speed", "wind_direction"]

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class Point(BaseModel):
    """A single geographic point with altitude. Used for all path coordinates."""

    model_config = ConfigDict(frozen=True)

    lat: float
    lon: float
    altitude_m: float


class BoidVelocity(BaseModel):
    model_config = ConfigDict(frozen=True)

    east: float
    north: float


class BoidAgent(BaseModel):
    agent_id: str
    position: Point
    velocity: BoidVelocity


class CandidatePath(BaseModel):
    path_id: str
    score: float
    points: list[Point]


class XaiResult(BaseModel):
    """Normalized (0-1) feature attribution values. Keys are AttributionFeatureKey."""

    attributions: dict[AttributionFeatureKey, float]  # type: ignore[valid-type]


# ---------------------------------------------------------------------------
# Server → Unity messages
# ---------------------------------------------------------------------------


class FrameMessage(BaseModel):
    """One frame sliced from a pre-computed prediction window."""

    schema_version: str = SCHEMA_VERSION
    message_type: Literal["frame"] = "frame"
    position: Point
    predicted_path: list[Point]
    candidates: list[CandidatePath]
    xai: XaiResult
    boids: list[BoidAgent] | None = None
    applied_overrides: dict[AttributionFeatureKey, float] | None = None  # type: ignore[valid-type]


class ErrorMessage(BaseModel):
    schema_version: str = SCHEMA_VERSION
    message_type: Literal["error"] = "error"
    code: Literal["startup_error", "bad_request", "pipeline_error"]
    detail: str


ServerMessage = Annotated[
    FrameMessage | ErrorMessage,
    Field(discriminator="message_type"),
]

# ---------------------------------------------------------------------------
# Unity → Server events
# ---------------------------------------------------------------------------


class ControlsSetEvent(BaseModel):
    schema_version: str
    message_type: Literal["controls.set"]
    overrides: dict[AttributionFeatureKey, float]  # type: ignore[valid-type]


_interaction_adapter: TypeAdapter[ControlsSetEvent] = TypeAdapter(ControlsSetEvent)


def parse_interaction_event(payload: dict) -> ControlsSetEvent:
    return _interaction_adapter.validate_python(payload)


# ---------------------------------------------------------------------------
# Internal pipeline contracts (never sent over the wire)
# ---------------------------------------------------------------------------


class ModelOutput(BaseModel):
    """Internal: predictor → server. Not sent to Unity."""

    contract_type: Literal["model.output"] = "model.output"
    predicted_window: list[Point]
    candidates: list[CandidatePath]
    raw_attributions: dict[AttributionFeatureKey, float]  # type: ignore[valid-type]
