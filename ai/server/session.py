"""Single-client session state."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai.common.models import AttributionFeatureKey


@dataclass
class SessionState:
    subject_id: str = "Osprey"
    overrides: dict[AttributionFeatureKey, float] = field(default_factory=dict)
