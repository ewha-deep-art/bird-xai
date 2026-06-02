"""Pydantic models for all Bird XAI contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"
AttributionFeatureKey = Literal["daylength_h", "ws_925", "q_850"]  # NOTE: XAI 대상이 되는 특성. 변경 가능
OverrideKey = Literal["message_cnt"] # NOTE: 인터랙션 입력 키

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class Point(BaseModel):
    """A single geographic point with altitude. Used for all path coordinates."""

    model_config = ConfigDict(frozen=True)

    lat: float
    lon: float
    altitude_m: float

# NOTE: Boids는 Unity에서 로컬 계산. 서버 frame은 항상 boids=null.
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
    applied_overrides: dict[OverrideKey, int] | None = None  # type: ignore[valid-type]


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
# Visitor → Server (HTTP)
# ---------------------------------------------------------------------------


class WishRequest(BaseModel):
    """POST /wish JSON body. Message text is not stored (v1)."""

    message: str = Field(..., min_length=1, max_length=50)


class WishResponse(BaseModel):
    status: Literal["ok"]
    backend: str
    subject_id: str