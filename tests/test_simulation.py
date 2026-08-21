from datetime import datetime, timezone
from pathlib import Path

import pytest
import pandas as pd

from src.avoidance import CollisionAvoidanceConfig, plan_avoidance_maneuvers, screen_conjunctions
from src.control import circular_orbit_raise_delta_v_m_s, propellant_required_kg
from src.models import StationKeepingConfig
from src.propagation import load_tles
from src.simulation import build_timestamps, simulate_tle_constellation


ROOT = Path(__file__).parents[1]
TLE_FILE = ROOT / "data" / "sample_constellation.tle"
OFFICIAL_TLE_FILE = ROOT / "data" / "official_fleet_10.tle"


def test_loads_five_satellites():
    assert len(load_tles(TLE_FILE)) == 5


def test_loads_ten_distinct_official_satellites():
    satellites = load_tles(OFFICIAL_TLE_FILE)
    assert len(satellites) == 10
    assert len({satellite.name for satellite in satellites}) == 10
    assert {"ISS (ZARYA)", "HST", "LANDSAT 9", "NOAA 20 (JPSS-1)"} <= {satellite.name for satellite in satellites}


def test_official_fleet_simulates_all_ten_satellites():
    telemetry = simulate_tle_constellation(
        OFFICIAL_TLE_FILE, datetime(2026, 8, 10, tzinfo=timezone.utc), duration_hours=1, step_seconds=1800,
    )
    assert len(telemetry) == 30
    assert telemetry["satellite"].nunique() == 10


def test_build_timestamps_is_inclusive():
    timestamps = build_timestamps(datetime(2026, 8, 9, tzinfo=timezone.utc), 1, 1800)
    assert len(timestamps) == 3
    assert timestamps[0].tzinfo is not None


def test_simulation_returns_ecef_and_altitude():
    telemetry = simulate_tle_constellation(
        TLE_FILE,
        start=datetime(2026, 8, 9, tzinfo=timezone.utc),
        duration_hours=1,
        step_seconds=1800,
    )
    assert len(telemetry) == 15
    assert {"ecef_x_km", "latitude_deg", "longitude_deg", "altitude_km"} <= set(telemetry.columns)
    assert telemetry["altitude_km"].between(300, 1_000).all()


def test_station_keeping_restores_altitude_and_consumes_fuel():
    telemetry = simulate_tle_constellation(
        TLE_FILE,
        start=datetime(2026, 8, 9, tzinfo=timezone.utc),
        duration_hours=24,
        step_seconds=3600,
        station_keeping=StationKeepingConfig(decay_rate_km_per_day=1.0, altitude_tolerance_km=0.2),
    )
    assert telemetry["station_keeping_event"].any()
    assert telemetry["fuel_remaining_kg"].min() < 2.0
    assert (telemetry.loc[telemetry["station_keeping_event"], "station_keeping_status"] == "MANEUVER").all()


def test_orbit_raise_and_rocket_equation_are_positive():
    config = StationKeepingConfig()
    delta_v = circular_orbit_raise_delta_v_m_s(400, 401)
    assert delta_v > 0
    assert propellant_required_kg(delta_v, 2.0, config) > 0
    with pytest.raises(ValueError):
        simulate_tle_constellation(
            TLE_FILE,
            datetime(2026, 8, 9, tzinfo=timezone.utc),
            station_keeping=StationKeepingConfig(altitude_tolerance_km=0),
        )


def test_conjunction_screen_finds_one_closest_approach_per_encounter():
    timestamps = pd.to_datetime(["2026-08-09T00:00:00Z", "2026-08-09T00:01:00Z", "2026-08-09T00:02:00Z"])
    telemetry = pd.DataFrame(
        [
            {"timestamp": timestamps[0], "satellite": "A", "ecef_x_km": 0, "ecef_y_km": 0, "ecef_z_km": 0},
            {"timestamp": timestamps[0], "satellite": "B", "ecef_x_km": 4, "ecef_y_km": 0, "ecef_z_km": 0},
            {"timestamp": timestamps[1], "satellite": "A", "ecef_x_km": 0, "ecef_y_km": 0, "ecef_z_km": 0},
            {"timestamp": timestamps[1], "satellite": "B", "ecef_x_km": 1, "ecef_y_km": 0, "ecef_z_km": 0},
            {"timestamp": timestamps[2], "satellite": "A", "ecef_x_km": 0, "ecef_y_km": 0, "ecef_z_km": 0},
            {"timestamp": timestamps[2], "satellite": "B", "ecef_x_km": 5, "ecef_y_km": 0, "ecef_z_km": 0},
        ]
    )
    config = CollisionAvoidanceConfig(keep_out_distance_km=2)
    events = screen_conjunctions(telemetry, config)
    plans = plan_avoidance_maneuvers(events, config)
    assert len(events) == 1
    assert events.iloc[0]["closest_approach_km"] == 1
    assert plans.iloc[0]["maneuver_satellite"] == "B"
    assert plans.iloc[0]["avoidance_delta_v_m_s"] > 0


def test_avoidance_plans_are_numerically_rescreened():
    telemetry = simulate_tle_constellation(
        TLE_FILE, datetime(2026, 8, 9, tzinfo=timezone.utc), duration_hours=2, step_seconds=600,
        station_keeping=StationKeepingConfig(), collision_avoidance=CollisionAvoidanceConfig(keep_out_distance_km=2_000),
    )
    maneuvers = telemetry[telemetry["avoidance_event"]]
    assert not maneuvers.empty
    assert maneuvers["predicted_post_burn_separation_km"].notna().all()
    assert set(maneuvers["avoidance_effectiveness"]) <= {"CLEARED", "INSUFFICIENT"}
    assert maneuvers["ai_risk_probability"].between(0, 1).all()
    assert set(maneuvers["ai_risk_label"]) <= {"LOW", "ELEVATED", "HIGH", "CRITICAL"}
    assert maneuvers["ai_primary_driver"].notna().all()
