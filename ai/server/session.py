"""Single-client session state."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai.common.constants import DEFAULT_SUBJECT_ID
from ai.contracts.models import AttributionFeatureKey


@dataclass
class SessionState:
    subject_id: str = DEFAULT_SUBJECT_ID
    overrides: dict[AttributionFeatureKey, float] = field(default_factory=dict)  # type: ignore[valid-type]
