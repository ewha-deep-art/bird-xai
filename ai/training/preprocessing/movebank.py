"""Movebank CSV normalization and resampling."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

from ai.common.constants import (
    MAX_FLIGHT_GAP_SECONDS,
    MAX_REASONABLE_SPEED_MPS,
    SAMPLE_DT_SECONDS,
)
from ai.common.geo import bearing_deg, estimate_speed_mps, haversine_distance_m
from ai.training.preprocessing.types import CanonicalRecord

MOVEbank_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def load_movebank_csv(path: str) -> list[CanonicalRecord]:
    records_by_subject: dict[str, list[CanonicalRecord]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row["location-lat"] or not row["location-long"]:
                continue
            timestamp = datetime.strptime(row["timestamp"], MOVEbank_TIMESTAMP_FORMAT).replace(
                tzinfo=timezone.utc
            )
            record = CanonicalRecord(
                event_id=row["event-id"],
                subject_id=row["individual-local-identifier"],
                timestamp=timestamp,
                lat=float(row["location-lat"]),
                lon=float(row["location-long"]),
                altitude_m=float(row.get("height-above-ellipsoid") or 0.0),
                ground_speed_mps=float(row.get("ground-speed") or 0.0),
                heading_deg=float(row.get("heading") or 0.0),
                segment_id="",
                gps_satellite_count=_safe_int(row.get("gps:satellite-count")),
                study_name=row.get("study-name") or "",
                study_timezone=row.get("study-timezone") or "UTC",
            )
            records_by_subject[record.subject_id].append(record)

    segmented: list[CanonicalRecord] = []
    for subject_id in sorted(records_by_subject):
        deduped = _deduplicate(records_by_subject[subject_id])
        for segment_index, segment in enumerate(split_on_large_gaps(deduped)):
            filtered = filter_track_records(segment)
            if len(filtered) < 2:
                continue
            smoothed = smooth_segment(filtered)
            resampled = resample_segment(smoothed)
            segment_id = f"{subject_id}-segment-{segment_index:04d}"
            for record in resampled:
                segmented.append(
                    CanonicalRecord(
                        **{
                            **record.__dict__,
                            "segment_id": segment_id,
                        }
                    )
                )
    return sorted(segmented, key=lambda item: (item.subject_id, item.timestamp))


def _safe_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _deduplicate(records: Iterable[CanonicalRecord]) -> list[CanonicalRecord]:
    ordered = sorted(records, key=lambda item: item.timestamp)
    deduped: dict[datetime, CanonicalRecord] = {}
    for record in ordered:
        deduped[record.timestamp] = record
    return sorted(deduped.values(), key=lambda item: item.timestamp)


def split_on_large_gaps(records: list[CanonicalRecord]) -> list[list[CanonicalRecord]]:
    if not records:
        return []
    segments: list[list[CanonicalRecord]] = [[records[0]]]
    for record in records[1:]:
        dt = (record.timestamp - segments[-1][-1].timestamp).total_seconds()
        if dt > MAX_FLIGHT_GAP_SECONDS:
            segments.append([record])
            continue
        segments[-1].append(record)
    return segments


def filter_track_records(records: list[CanonicalRecord]) -> list[CanonicalRecord]:
    filtered: list[CanonicalRecord] = []
    previous: CanonicalRecord | None = None
    for record in records:
        if previous is not None:
            dt = (record.timestamp - previous.timestamp).total_seconds()
            distance = haversine_distance_m(previous.lat, previous.lon, record.lat, record.lon)
            estimated_speed = estimate_speed_mps(distance, dt)
            if estimated_speed > MAX_REASONABLE_SPEED_MPS:
                continue
        filtered.append(record)
        previous = record
    return filtered


def smooth_segment(records: list[CanonicalRecord]) -> list[CanonicalRecord]:
    latitudes = _kalman_smooth([record.lat for record in records], process_variance=1e-6)
    longitudes = _kalman_smooth([record.lon for record in records], process_variance=1e-6)
    altitudes = _kalman_smooth([record.altitude_m for record in records], process_variance=0.2)

    smoothed: list[CanonicalRecord] = []
    for index, record in enumerate(records):
        smoothed.append(
            CanonicalRecord(
                **{
                    **record.__dict__,
                    "lat": latitudes[index],
                    "lon": longitudes[index],
                    "altitude_m": altitudes[index],
                }
            )
        )
    return smoothed


def _kalman_smooth(
    values: list[float],
    process_variance: float,
    measurement_variance: float = 1e-3,
) -> list[float]:
    if not values:
        return []
    estimate = values[0]
    error_estimate = 1.0
    output = []
    for measurement in values:
        error_estimate += process_variance
        gain = error_estimate / (error_estimate + measurement_variance)
        estimate += gain * (measurement - estimate)
        error_estimate *= 1.0 - gain
        output.append(estimate)
    return output


def resample_segment(records: list[CanonicalRecord]) -> list[CanonicalRecord]:
    if len(records) < 2:
        return records
    try:
        from scipy.interpolate import CubicSpline  # type: ignore
    except Exception:
        CubicSpline = None

    times = [record.timestamp for record in records]
    offsets = [(timestamp - times[0]).total_seconds() for timestamp in times]
    first = _ceil_timestamp(times[0], SAMPLE_DT_SECONDS)
    last = _floor_timestamp(times[-1], SAMPLE_DT_SECONDS)
    if first > last:
        return records
    target_times: list[datetime] = []
    cursor = first
    while cursor <= last:
        target_times.append(cursor)
        cursor += timedelta(seconds=SAMPLE_DT_SECONDS)
    target_offsets = [(timestamp - times[0]).total_seconds() for timestamp in target_times]

    latitudes = _interpolate_series(offsets, [record.lat for record in records], target_offsets, CubicSpline)
    longitudes = _interpolate_series(offsets, [record.lon for record in records], target_offsets, CubicSpline)
    altitudes = _interpolate_series(
        offsets, [record.altitude_m for record in records], target_offsets, CubicSpline
    )

    resampled: list[CanonicalRecord] = []
    previous: CanonicalRecord | None = None
    for index, timestamp in enumerate(target_times):
        lat = latitudes[index]
        lon = longitudes[index]
        altitude = altitudes[index]
        if previous is None:
            speed = records[0].ground_speed_mps
            heading = records[0].heading_deg
        else:
            distance = haversine_distance_m(previous.lat, previous.lon, lat, lon)
            speed = estimate_speed_mps(distance, SAMPLE_DT_SECONDS)
            heading = bearing_deg(previous.lat, previous.lon, lat, lon)
        resampled_record = CanonicalRecord(
            event_id=f"{records[0].subject_id}-{timestamp.isoformat()}",
            subject_id=records[0].subject_id,
            timestamp=timestamp,
            lat=lat,
            lon=lon,
            altitude_m=altitude,
            ground_speed_mps=speed,
            heading_deg=heading,
            segment_id=records[0].segment_id,
            gps_satellite_count=records[0].gps_satellite_count,
            study_name=records[0].study_name,
            study_timezone=records[0].study_timezone,
        )
        resampled.append(resampled_record)
        previous = resampled_record
    return resampled


def _ceil_timestamp(timestamp: datetime, seconds: int) -> datetime:
    epoch = int(timestamp.timestamp())
    remainder = epoch % seconds
    if remainder == 0:
        return timestamp
    return datetime.fromtimestamp(epoch + (seconds - remainder), tz=timestamp.tzinfo)


def _floor_timestamp(timestamp: datetime, seconds: int) -> datetime:
    epoch = int(timestamp.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % seconds), tz=timestamp.tzinfo)


def _interpolate_series(
    source_offsets: list[float],
    values: list[float],
    target_offsets: list[float],
    cubic_spline_cls,
) -> list[float]:
    if cubic_spline_cls is not None and len(source_offsets) >= 4:
        spline = cubic_spline_cls(source_offsets, values)
        return [float(spline(offset)) for offset in target_offsets]
    return [_linear_interpolate(source_offsets, values, target) for target in target_offsets]


def _linear_interpolate(source_offsets: list[float], values: list[float], target: float) -> float:
    if target <= source_offsets[0]:
        return values[0]
    if target >= source_offsets[-1]:
        return values[-1]
    for index in range(1, len(source_offsets)):
        left_offset = source_offsets[index - 1]
        right_offset = source_offsets[index]
        if target > right_offset:
            continue
        fraction = (target - left_offset) / (right_offset - left_offset)
        return values[index - 1] + (values[index] - values[index - 1]) * fraction
    return values[-1]
