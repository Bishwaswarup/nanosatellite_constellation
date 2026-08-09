"""Simulation orchestration for a TLE-defined satellite constellation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .control import advance_station_keeping, initialize_state
from .models import StationKeepingConfig
from .propagation import load_tles, propagate_constellation


def build_timestamps(start: datetime, duration_hours: int, step_seconds: int) -> list[datetime]:
    """Create inclusive, timezone-aware simulation timestamps."""
    if start.tzinfo is None:
        raise ValueError("Simulation start time must be timezone-aware.")
    if duration_hours <= 0 or step_seconds <= 0:
        raise ValueError("Duration and step size must be positive.")
    start = start.astimezone(timezone.utc)
    steps = duration_hours * 3600 // step_seconds
    return [start + timedelta(seconds=step * step_seconds) for step in range(steps + 1)]


def simulate_tle_constellation(
    tle_path: str | Path,
    start: datetime,
    duration_hours: int = 24,
    step_seconds: int = 300,
    station_keeping: StationKeepingConfig | None = None,
) -> pd.DataFrame:
    """Generate SGP4 telemetry and optional simplified station-keeping state."""
    satellites = load_tles(tle_path)
    timestamps = build_timestamps(start, duration_hours, step_seconds)
    telemetry = propagate_constellation(satellites, timestamps)
    frame = pd.DataFrame(telemetry).sort_values(["satellite", "timestamp"]).reset_index(drop=True)
    if station_keeping is None:
        return frame.sort_values(["timestamp", "satellite"]).reset_index(drop=True)

    controlled_rows: list[dict[str, object]] = []
    for _, satellite_rows in frame.groupby("satellite", sort=False):
        state = initialize_state(float(satellite_rows.iloc[0]["mean_orbit_altitude_km"]), station_keeping)
        previous_timestamp = None
        for _, row in satellite_rows.iterrows():
            timestamp = row["timestamp"]
            elapsed_seconds = 0.0 if previous_timestamp is None else (timestamp - previous_timestamp).total_seconds()
            controlled_rows.append({**row.to_dict(), **advance_station_keeping(state, elapsed_seconds, station_keeping)})
            previous_timestamp = timestamp
    return pd.DataFrame(controlled_rows).sort_values(["timestamp", "satellite"]).reset_index(drop=True)
