"""Simulation orchestration for a TLE-defined satellite constellation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .avoidance import CollisionAvoidanceConfig, plan_avoidance_maneuvers, screen_conjunctions
from .control import advance_station_keeping, initialize_state, propellant_required_kg
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
    collision_avoidance: CollisionAvoidanceConfig | None = None,
) -> pd.DataFrame:
    """Generate SGP4 telemetry plus optional station-keeping and avoidance state."""
    satellites = load_tles(tle_path)
    timestamps = build_timestamps(start, duration_hours, step_seconds)
    telemetry = propagate_constellation(satellites, timestamps)
    frame = pd.DataFrame(telemetry).sort_values(["satellite", "timestamp"]).reset_index(drop=True)
    if station_keeping is None and collision_avoidance is None:
        return frame.sort_values(["timestamp", "satellite"]).reset_index(drop=True)

    control_config = station_keeping or StationKeepingConfig()
    plans = plan_avoidance_maneuvers(screen_conjunctions(frame, collision_avoidance), collision_avoidance) if collision_avoidance else pd.DataFrame()
    alert_map: dict[tuple[object, str], dict[str, object]] = {}
    maneuver_map: dict[tuple[object, str], dict[str, object]] = {}
    for _, plan in plans.iterrows():
        for satellite in (plan["satellite_a"], plan["satellite_b"]):
            alert_map[(plan["timestamp"], satellite)] = plan.to_dict()
        maneuver_map[(plan["timestamp"], plan["maneuver_satellite"])] = plan.to_dict()

    controlled_rows: list[dict[str, object]] = []
    for _, satellite_rows in frame.groupby("satellite", sort=False):
        state = initialize_state(float(satellite_rows.iloc[0]["mean_orbit_altitude_km"]), control_config)
        previous_timestamp = None
        for _, row in satellite_rows.iterrows():
            timestamp = row["timestamp"]
            elapsed_seconds = 0.0 if previous_timestamp is None else (timestamp - previous_timestamp).total_seconds()
            result = advance_station_keeping(state, elapsed_seconds, control_config)
            alert = alert_map.get((timestamp, row["satellite"]))
            plan = maneuver_map.get((timestamp, row["satellite"]))
            result.update(
                {
                    "conjunction_alert": alert is not None,
                    "closest_approach_km": float(alert["closest_approach_km"]) if alert else None,
                    "avoidance_event": False,
                    "avoidance_delta_v_m_s": 0.0,
                    "avoidance_status": "CLEAR",
                }
            )
            if plan is not None:
                delta_v = float(plan["avoidance_delta_v_m_s"])
                propellant_used = propellant_required_kg(delta_v, state.propellant_remaining_kg, control_config)
                if propellant_used <= state.propellant_remaining_kg:
                    state.propellant_remaining_kg -= propellant_used
                    state.cumulative_delta_v_m_s += delta_v
                    result.update(
                        {
                            "fuel_remaining_kg": state.propellant_remaining_kg,
                            "fuel_percent": 100 * state.propellant_remaining_kg / control_config.initial_propellant_kg,
                            "cumulative_delta_v_m_s": state.cumulative_delta_v_m_s,
                            "avoidance_event": True,
                            "avoidance_delta_v_m_s": delta_v,
                            "avoidance_status": "MANEUVER_PLANNED",
                        }
                    )
                else:
                    result["avoidance_status"] = "OUT_OF_FUEL"
            controlled_rows.append({**row.to_dict(), **result})
            previous_timestamp = timestamp
    return pd.DataFrame(controlled_rows).sort_values(["timestamp", "satellite"]).reset_index(drop=True)
