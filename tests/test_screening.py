"""Tests for between-sample conjunction screening and the differential post-burn check."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.avoidance import (CollisionAvoidanceConfig, _golden_minimum, closest_approaches,
                           hermite_min_separation, screen_conjunctions)
from src.models import StationKeepingConfig
from src.propagation import load_tles, make_state_function, tle_epoch
from src.simulation import simulate_tle_constellation

ROOT = Path(__file__).parents[1]
FLEET = ROOT / "data" / "official_fleet_10.tle"


def test_hermite_is_exact_for_straight_line_crossing():
    # relative motion r(t) = (1, -7.5 + 0.05 t, 0) km over 300 s: miss 1 km at t = 150 s
    r0, v = np.array([[1.0, -7.5, 0.0]]), np.array([[0.0, 0.05, 0.0]])
    d, s = hermite_min_separation(r0, v, r0 + 300 * v, v, 300.0, n_sub=301)
    assert abs(d[0] - 1.0) < 1e-9
    assert abs(s[0] - 0.5) < 1e-6


def test_golden_section_finds_minimum():
    x, f = _golden_minimum(lambda t: (t - 3.3) ** 2 + 2.0, 0.0, 10.0, tol=1e-6)
    assert abs(x - 3.3) < 1e-4 and abs(f - 2.0) < 1e-8


def test_screen_catches_encounter_missed_between_samples():
    """Two synthetic objects cross 1 km apart midway between samples 5 min apart."""
    t = pd.to_datetime(["2026-08-09T00:00:00Z", "2026-08-09T00:05:00Z"])
    rows = []
    for k, tk in enumerate(t):
        ts = k * 300.0
        rows.append({"timestamp": tk, "satellite": "A", "ecef_x_km": 0, "ecef_y_km": 0, "ecef_z_km": 0,
                     "teme_x_km": 0, "teme_y_km": 0, "teme_z_km": 0,
                     "teme_vx_km_s": 0, "teme_vy_km_s": 0, "teme_vz_km_s": 0})
        y = -1500 + 10 * ts
        rows.append({"timestamp": tk, "satellite": "B", "ecef_x_km": 1, "ecef_y_km": y, "ecef_z_km": 0,
                     "teme_x_km": 1, "teme_y_km": y, "teme_z_km": 0,
                     "teme_vx_km_s": 0, "teme_vy_km_s": 10, "teme_vz_km_s": 0})
    frame = pd.DataFrame(rows)
    cfg = CollisionAvoidanceConfig(keep_out_distance_km=2.0)
    sampled_only = screen_conjunctions(frame.drop(columns=[c for c in frame if c.startswith("teme_")]), cfg)
    hermite = screen_conjunctions(frame, cfg)
    assert sampled_only.empty                            # 1500 km apart at both samples
    assert len(hermite) == 1
    assert abs(hermite.iloc[0]["closest_approach_km"] - 1.0) < 1e-3
    assert abs(hermite.iloc[0]["time_to_closest_approach_s"] - 150.0) < 5.0


def test_tle_epoch():
    iss = load_tles(FLEET)[0]
    assert tle_epoch(iss).date().isoformat() == "2026-08-08"


@pytest.fixture(scope="module")
def fleet_day():
    sats = load_tles(FLEET)
    start = max(tle_epoch(s) for s in sats).replace(microsecond=0)
    return sats, start, simulate_tle_constellation(FLEET, start, duration_hours=24, step_seconds=300)


def test_refined_closest_approach_matches_brute_force(fleet_day):
    sats, _, frame = fleet_day
    fn = make_state_function(sats)
    best = closest_approaches(frame, fn).iloc[0]
    a, b, tca = best["satellite_a"], best["satellite_b"], best["tca_time"]
    brute = min(np.linalg.norm(fn(a, tca + timedelta(seconds=x))[0] - fn(b, tca + timedelta(seconds=x))[0])
                for x in np.arange(-60, 60, 0.25))
    assert best["closest_approach_km"] <= brute + 0.01
    assert best["closest_approach_km"] < 40                # the sampled screen never sees this pass


def test_zero_burn_reproduces_sgp4_miss_distance(fleet_day):
    sats, start, _ = fleet_day
    cfg = CollisionAvoidanceConfig(keep_out_distance_km=50.0, minimum_delta_v_m_s=1e-9, maximum_delta_v_m_s=1e-9)
    telemetry = simulate_tle_constellation(FLEET, start, 24, 300, station_keeping=StationKeepingConfig(),
                                           collision_avoidance=cfg)
    m = telemetry[telemetry["avoidance_event"]]
    alerts = telemetry[telemetry["conjunction_alert"]]
    assert not m.empty
    for _, row in m.iterrows():
        miss = alerts["closest_approach_km"].dropna()
        assert np.min(np.abs(miss.to_numpy() - row["predicted_post_burn_separation_km"])) < 1e-3
