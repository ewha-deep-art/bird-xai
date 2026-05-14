"""Dataclasses shared by preprocessing, inference, and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import cos, pi, sin

from ai.common.constants import MIN_FLIGHT_SPEED_MPS
from ai.common.geo import heading_components


@dataclass(frozen=True)
class CanonicalRecord:
    event_id: str
    subject_id: str
    timestamp: datetime
    lat: float
    lon: float
    altitude_m: float
    ground_speed_mps: float
    heading_deg: float
    segment_id: str
    gps_satellite_count: int | None = None
    study_name: str = ""
    study_timezone: str = "UTC"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "CanonicalRecord":
        return cls(
            event_id=payload["event_id"],
            subject_id=payload["subject_id"],
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            lat=float(payload["lat"]),
            lon=float(payload["lon"]),
            altitude_m=float(payload["altitude_m"]),
            ground_speed_mps=float(payload["ground_speed_mps"]),
            heading_deg=float(payload["heading_deg"]),
            segment_id=payload["segment_id"],
            gps_satellite_count=payload.get("gps_satellite_count"),
            study_name=payload.get("study_name", ""),
            study_timezone=payload.get("study_timezone", "UTC"),
        )


@dataclass(frozen=True)
class EnrichedRecord:
    event_id: str
    subject_id: str
    timestamp: datetime
    lat: float
    lon: float
    altitude_m: float
    ground_speed_mps: float
    heading_deg: float
    heading_sin: float
    heading_cos: float
    eastward_wind: float
    northward_wind: float
    vertical_velocity: float
    wind_speed: float
    day_of_year_sin: float
    day_of_year_cos: float
    hour_of_day_sin: float
    hour_of_day_cos: float
    is_in_flight: float
    pressure_level_hpa: int
    segment_id: str
    era5_source_name: str

    @classmethod
    def from_canonical(
        cls,
        canonical: CanonicalRecord,
        eastward_wind: float,
        northward_wind: float,
        vertical_velocity: float,
        wind_speed: float,
        pressure_level_hpa: int,
        era5_source_name: str,
    ) -> "EnrichedRecord":
        heading_sin, heading_cos = heading_components(canonical.heading_deg)
        day_of_year_sin, day_of_year_cos = _day_of_year_components(canonical.timestamp)
        hour_of_day_sin, hour_of_day_cos = _hour_of_day_components(canonical.timestamp)
        return cls(
            event_id=canonical.event_id,
            subject_id=canonical.subject_id,
            timestamp=canonical.timestamp,
            lat=canonical.lat,
            lon=canonical.lon,
            altitude_m=canonical.altitude_m,
            ground_speed_mps=canonical.ground_speed_mps,
            heading_deg=canonical.heading_deg,
            heading_sin=heading_sin,
            heading_cos=heading_cos,
            eastward_wind=eastward_wind,
            northward_wind=northward_wind,
            vertical_velocity=vertical_velocity,
            wind_speed=wind_speed,
            day_of_year_sin=day_of_year_sin,
            day_of_year_cos=day_of_year_cos,
            hour_of_day_sin=hour_of_day_sin,
            hour_of_day_cos=hour_of_day_cos,
            is_in_flight=1.0 if canonical.ground_speed_mps > MIN_FLIGHT_SPEED_MPS else 0.0,
            pressure_level_hpa=pressure_level_hpa,
            segment_id=canonical.segment_id,
            era5_source_name=era5_source_name,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "EnrichedRecord":
        timestamp = datetime.fromisoformat(payload["timestamp"])
        day_of_year_sin, day_of_year_cos = _day_of_year_components(timestamp)
        hour_of_day_sin, hour_of_day_cos = _hour_of_day_components(timestamp)
        ground_speed = float(payload["ground_speed_mps"])
        return cls(
            event_id=payload["event_id"],
            subject_id=payload["subject_id"],
            timestamp=timestamp,
            lat=float(payload["lat"]),
            lon=float(payload["lon"]),
            altitude_m=float(payload["altitude_m"]),
            ground_speed_mps=ground_speed,
            heading_deg=float(payload["heading_deg"]),
            heading_sin=float(payload["heading_sin"]),
            heading_cos=float(payload["heading_cos"]),
            eastward_wind=float(payload["eastward_wind"]),
            northward_wind=float(payload["northward_wind"]),
            vertical_velocity=float(payload["vertical_velocity"]),
            wind_speed=float(payload["wind_speed"]),
            day_of_year_sin=float(payload.get("day_of_year_sin", day_of_year_sin)),
            day_of_year_cos=float(payload.get("day_of_year_cos", day_of_year_cos)),
            hour_of_day_sin=float(payload.get("hour_of_day_sin", hour_of_day_sin)),
            hour_of_day_cos=float(payload.get("hour_of_day_cos", hour_of_day_cos)),
            is_in_flight=float(
                payload.get("is_in_flight", 1.0 if ground_speed > MIN_FLIGHT_SPEED_MPS else 0.0)
            ),
            pressure_level_hpa=int(payload["pressure_level_hpa"]),
            segment_id=payload["segment_id"],
            era5_source_name=payload["era5_source_name"],
        )


def _day_of_year_components(timestamp: datetime) -> tuple[float, float]:
    day_of_year = max(1, min(timestamp.timetuple().tm_yday, 366))
    angle = 2.0 * pi * ((day_of_year - 1) / 366.0)
    return sin(angle), cos(angle)


def _hour_of_day_components(timestamp: datetime) -> tuple[float, float]:
    hour = timestamp.hour + (timestamp.minute / 60.0) + (timestamp.second / 3600.0)
    angle = 2.0 * pi * (hour / 24.0)
    return sin(angle), cos(angle)
