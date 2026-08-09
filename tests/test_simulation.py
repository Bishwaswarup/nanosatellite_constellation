from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.control import circular_orbit_raise_delta_v_m_s, propellant_required_kg
from src.models import StationKeepingConfig
from src.propagation import load_tles
from src.simulation import build_timestamps, simulate_tle_constellation


ROOT = Path(__file__).parents[1]
TLE_FILE = ROOT / "data" / "sample_constellation.tle"


def test_loads_five_satellites():
    assert len(load_tles(TLE_FILE)) == 5


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
