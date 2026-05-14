"""Preprocessing pipeline entrypoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from ai.common.artifacts import ArtifactMetadata, load_json_artifact, write_json_artifact
from ai.common.constants import (
    MODEL_INPUT_FEATURE_ORDER,
    OBSERVED_STEPS,
    PREDICTED_STEPS,
    SAMPLE_DT_SECONDS,
    SCHEMA_VERSION,
)
from ai.config import get_settings
from ai.training.model.dataset import record_to_feature_row
from ai.training.preprocessing.era5 import Era5Mapper
from ai.training.preprocessing.movebank import load_movebank_csv
from ai.training.preprocessing.types import EnrichedRecord


def build_runtime_records(
    movebank_path: str | Path | None = None,
    era5_path: str | Path | None = None,
) -> list[EnrichedRecord]:
    settings = get_settings()
    canonical_records = load_movebank_csv(str(movebank_path or settings.raw_movebank_path))
    mapper = Era5Mapper(str(era5_path or settings.raw_era5_path))
    enriched_records = [
        EnrichedRecord.from_canonical(
            canonical=record,
            eastward_wind=features.eastward_wind,
            northward_wind=features.northward_wind,
            vertical_velocity=features.vertical_velocity,
            wind_speed=features.wind_speed,
            pressure_level_hpa=features.pressure_level_hpa,
            era5_source_name=features.source_name,
        )
        for record in canonical_records
        for features in [mapper.map_record(record)]
    ]
    return enriched_records


def build_runtime_records_with_metadata(
    movebank_path: str | Path | None = None,
    era5_path: str | Path | None = None,
) -> tuple[ArtifactMetadata, list[EnrichedRecord]]:
    records = build_runtime_records(movebank_path, era5_path)
    source_names = sorted({record.era5_source_name for record in records}) if records else []
    metadata = build_processed_metadata(
        records=records,
        artifact_type="records",
        window_count=None,
        source_name=",".join(source_names) if source_names else "",
    )
    return metadata, records


def build_dataset_windows(
    records: list[EnrichedRecord],
    observed_steps: int = OBSERVED_STEPS,
    predicted_steps: int = PREDICTED_STEPS,
) -> list[dict]:
    windows: list[dict] = []
    by_segment: dict[str, list[EnrichedRecord]] = {}
    for record in records:
        by_segment.setdefault(record.segment_id, []).append(record)

    for segment_id, segment_records in by_segment.items():
        ordered = sorted(segment_records, key=lambda item: item.timestamp)
        window_size = observed_steps + predicted_steps
        if len(ordered) < window_size:
            continue
        for start in range(0, len(ordered) - window_size + 1):
            observed = ordered[start : start + observed_steps]
            predicted = ordered[start + observed_steps : start + window_size]
            windows.append(
                {
                    "subject_id": observed[-1].subject_id,
                    "segment_id": segment_id,
                    "observed_start": observed[0].timestamp.isoformat(),
                    "observed_end": observed[-1].timestamp.isoformat(),
                    "anchor_time": observed[-1].timestamp.isoformat(),
                    "predicted_end": predicted[-1].timestamp.isoformat(),
                    "input_features": _build_feature_matrix(observed),
                    "target_deltas": _build_target_matrix(observed[-1], predicted),
                }
            )
    return windows


def _build_feature_matrix(records: list[EnrichedRecord]) -> list[list[float]]:
    matrix: list[list[float]] = []
    previous = None
    for record in records:
        matrix.append(record_to_feature_row(record, previous))
        previous = record
    return matrix


def _build_target_matrix(anchor: EnrichedRecord, predicted: list[EnrichedRecord]) -> list[list[float]]:
    matrix: list[list[float]] = []
    previous_lat = anchor.lat
    previous_lon = anchor.lon
    previous_altitude = anchor.altitude_m
    for record in predicted:
        matrix.append(
            [
                record.lat - previous_lat,
                record.lon - previous_lon,
                record.altitude_m - previous_altitude,
            ]
        )
        previous_lat = record.lat
        previous_lon = record.lon
        previous_altitude = record.altitude_m
    return matrix


def build_processed_metadata(
    *,
    records: list[EnrichedRecord],
    artifact_type: str,
    window_count: int | None,
    source_name: str,
) -> ArtifactMetadata:
    return ArtifactMetadata(
        artifact_type=artifact_type,
        schema_version=SCHEMA_VERSION,
        subject_id=records[0].subject_id if records else "",
        feature_order=MODEL_INPUT_FEATURE_ORDER,
        strict_era5=True,
        record_count=len(records),
        segment_count=len(Counter(record.segment_id for record in records)),
        window_count=window_count,
        source_name=source_name,
    )


def load_enriched_record_artifact(
    path: str | Path | None = None,
    *,
    require_metadata: bool = True,
) -> tuple[ArtifactMetadata, list[EnrichedRecord]]:
    settings = get_settings()
    metadata, items = load_json_artifact(
        path or settings.features_artifact_path,
        items_key="records",
        require_metadata=require_metadata,
    )
    return metadata, [EnrichedRecord.from_dict(item) for item in items]


def run_preprocessing() -> dict[str, Path]:
    settings = get_settings()
    settings.canonical_root.mkdir(parents=True, exist_ok=True)
    settings.features_root.mkdir(parents=True, exist_ok=True)
    settings.datasets_root.mkdir(parents=True, exist_ok=True)

    records_metadata, records = build_runtime_records_with_metadata()
    windows = build_dataset_windows(records)
    canonical_path = settings.canonical_artifact_path
    features_path = settings.features_artifact_path
    dataset_path = settings.dataset_artifact_path

    canonical_metadata = ArtifactMetadata(
        artifact_type="canonical",
        schema_version=records_metadata.schema_version,
        subject_id=records_metadata.subject_id,
        feature_order=records_metadata.feature_order,
        strict_era5=records_metadata.strict_era5,
        record_count=records_metadata.record_count,
        segment_count=records_metadata.segment_count,
        source_name=records_metadata.source_name,
    )
    features_metadata = ArtifactMetadata(
        artifact_type="records",
        schema_version=records_metadata.schema_version,
        subject_id=records_metadata.subject_id,
        feature_order=records_metadata.feature_order,
        strict_era5=records_metadata.strict_era5,
        record_count=records_metadata.record_count,
        segment_count=records_metadata.segment_count,
        source_name=records_metadata.source_name,
    )
    dataset_metadata = ArtifactMetadata(
        artifact_type="windows",
        schema_version=records_metadata.schema_version,
        subject_id=records_metadata.subject_id,
        feature_order=records_metadata.feature_order,
        strict_era5=records_metadata.strict_era5,
        record_count=records_metadata.record_count,
        segment_count=records_metadata.segment_count,
        window_count=len(windows),
        source_name=records_metadata.source_name,
    )

    write_json_artifact(
        canonical_path,
        metadata=canonical_metadata,
        items_key="records",
        items=[_canonical_projection(record) for record in records],
    )
    write_json_artifact(
        features_path,
        metadata=features_metadata,
        items_key="records",
        items=[record.to_dict() for record in records],
    )
    write_json_artifact(
        dataset_path,
        metadata=dataset_metadata,
        items_key="windows",
        items=windows,
    )
    return {
        "canonical": canonical_path,
        "features": features_path,
        "datasets": dataset_path,
    }


def _canonical_projection(record: EnrichedRecord) -> dict:
    base = {
        "event_id": record.event_id,
        "subject_id": record.subject_id,
        "timestamp": record.timestamp.isoformat(),
        "lat": record.lat,
        "lon": record.lon,
        "altitude_m": record.altitude_m,
        "ground_speed_mps": record.ground_speed_mps,
        "heading_deg": record.heading_deg,
        "segment_id": record.segment_id,
    }
    return base


def main() -> None:
    argparse.ArgumentParser(description="Run Bird XAI preprocessing").parse_args()
    outputs = run_preprocessing()
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
