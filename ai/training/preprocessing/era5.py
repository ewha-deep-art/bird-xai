"""ERA5 nearest-neighbor mapping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from math import sqrt

from ai.common.constants import DEFAULT_PRESSURE_LEVEL_HPA
from ai.training.preprocessing.types import CanonicalRecord


@dataclass(frozen=True)
class Era5Features:
    eastward_wind: float
    northward_wind: float
    vertical_velocity: float
    wind_speed: float
    pressure_level_hpa: int
    source_name: str


class Era5Mapper:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self._dataset = None
        self._xarray = None
        self._variable_map: dict[str, str] = {}
        try:
            import xarray as xr  # type: ignore
        except Exception:
            xr = None
        if xr is not None:
            try:
                self._xarray = xr
                self._dataset = xr.open_dataset(dataset_path)
                self._variable_map = self._detect_variables()
            except Exception:
                self._dataset = None
                self._xarray = None
                self._variable_map = {}

    def map_record(self, record: CanonicalRecord) -> Era5Features:
        if self._dataset is None or self._xarray is None:
            raise RuntimeError(
                f"ERA5 mapping requires a readable xarray dataset and detected variables at {self.dataset_path}."
            )
        try:
            return self._map_record_with_dataset(record)
        except Exception as exc:
            raise RuntimeError(
                f"ERA5 mapping failed while reading record at {record.timestamp.isoformat()}."
            ) from exc

    def _map_record_with_dataset(self, record: CanonicalRecord) -> Era5Features:
        dataset = self._dataset
        assert dataset is not None
        time_coord = "valid_time" if "valid_time" in dataset.coords else "time"
        lat_coord = "latitude" if "latitude" in dataset.coords else "lat"
        lon_coord = "longitude" if "longitude" in dataset.coords else "lon"
        pressure_coord = None
        for candidate in ("pressure_level", "isobaricInhPa", "level"):
            if candidate in dataset.coords:
                pressure_coord = candidate
                break

        selection = {
            time_coord: record.timestamp.astimezone(timezone.utc).replace(tzinfo=None),
            lat_coord: record.lat,
            lon_coord: record.lon % 360 if float(dataset[lon_coord].max()) > 180 else record.lon,
        }

        selected_pressure = DEFAULT_PRESSURE_LEVEL_HPA
        if pressure_coord is not None:
            levels = [int(level) for level in dataset[pressure_coord].values.tolist()]
            selected_pressure = _select_pressure_level(record.altitude_m, levels)
            selection[pressure_coord] = selected_pressure

        subset = dataset.sel(selection, method="nearest")
        eastward = float(subset[self._variable_map["eastward_wind"]].item())
        northward = float(subset[self._variable_map["northward_wind"]].item())
        vertical_velocity = float(subset[self._variable_map["vertical_velocity"]].item())
        return Era5Features(
            eastward_wind=eastward,
            northward_wind=northward,
            vertical_velocity=vertical_velocity,
            wind_speed=sqrt(eastward**2 + northward**2),
            pressure_level_hpa=selected_pressure,
            source_name="era5",
        )

    def _detect_variables(self) -> dict[str, str]:
        assert self._dataset is not None
        variables = list(self._dataset.data_vars)
        return {
            "eastward_wind": _match_variable_name(
                variables,
                ("eastward_wind", "u", "u_component_of_wind", "u_component"),
            ),
            "northward_wind": _match_variable_name(
                variables,
                ("northward_wind", "v", "v_component_of_wind", "v_component"),
            ),
            "vertical_velocity": _match_variable_name(
                variables,
                ("vertical_velocity", "lagrangian_tendency_of_air_pressure", "w"),
            ),
        }

def _match_variable_name(variables: list[str], candidates: tuple[str, ...]) -> str:
    lowered = {variable.lower(): variable for variable in variables}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for candidate in candidates:
        for variable in variables:
            if candidate.lower() in variable.lower():
                return variable
    raise KeyError(f"Could not match ERA5 variable for {candidates}")


def _select_pressure_level(altitude_m: float, available_levels: list[int]) -> int:
    if not available_levels:
        return DEFAULT_PRESSURE_LEVEL_HPA
    if altitude_m < 0:
        return DEFAULT_PRESSURE_LEVEL_HPA
    estimated_pressure = 1013.25 * max(0.01, (1.0 - altitude_m / 44330.0)) ** 5.255
    return min(available_levels, key=lambda level: abs(level - estimated_pressure))
