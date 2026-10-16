"""Regenerate figures/ and results/report.json for the README.

    python scripts/make_report.py      # ~1 min

Uses the archived 10-satellite fleet, starting at its TLE epoch, for 24 h.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.avoidance import (CollisionAvoidanceConfig, _screen_sampled, closest_approaches,  # noqa: E402
                           hermite_min_separation)
from src.models import StationKeepingConfig  # noqa: E402
from src.numerical import apply_velocity_burn, propagate_two_body  # noqa: E402
from src.propagation import load_tles, make_state_function, tle_epoch  # noqa: E402
from src.simulation import simulate_tle_constellation  # noqa: E402

FLEET = ROOT / "data" / "official_fleet_10.tle"
FIG, RES = ROOT / "figures", ROOT / "results"
BLUE, ORANGE, AQUA, MUTED, INK = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#0b0b0b"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e4e3df", "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False,
})


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    sats = load_tles(FLEET)
    fn = make_state_function(sats)
    start = max(tle_epoch(s) for s in sats).replace(microsecond=0)
    report: dict = {"start_utc": start.isoformat(), "duration_h": 24, "satellites": [s.name for s in sats]}

    # ── closest approaches ─────────────────────────────────────────────
    frames = {step: simulate_tle_constellation(FLEET, start, 24, step) for step in (300, 60)}
    exact = closest_approaches(frames[300], fn)
    report["closest_approaches"] = [
        {**{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}}
        for row in exact.head(10).to_dict("records")
    ]
    sampled = {}
    for step, fr in frames.items():
        ev = _screen_sampled(fr, CollisionAvoidanceConfig(keep_out_distance_km=1e9))
        g = ev.groupby(["satellite_a", "satellite_b"])["closest_approach_km"].min()
        sampled[step] = g
    best = exact.iloc[0]
    key = (best["satellite_a"], best["satellite_b"])
    report["best_pair"] = {
        "pair": list(key), "exact_km": float(best["closest_approach_km"]),
        "tca_utc": best["tca_time"].isoformat(), "relative_speed_km_s": float(best["relative_velocity_km_s"]),
        "sampled_300s_min_over_run_km": float(sampled[300][key]), "sampled_60s_min_over_run_km": float(sampled[60][key]),
    }
    herm = closest_approaches(frames[300]).set_index(["satellite_a", "satellite_b"])["closest_approach_km"]
    ex = exact.set_index(["satellite_a", "satellite_b"])["closest_approach_km"]
    report["hermite_max_error_km"] = float((herm - ex.reindex(herm.index)).abs().max())
    report["sampled_300s_misses_below_50km"] = int((ex < 50).sum() - (sampled[300].reindex(ex.index) < 50).sum())

    # Figure 1: why sampling misses the pass
    a, b, tca = key[0], key[1], best["tca_time"]
    xs = np.arange(-900, 900, 1.0)
    true = np.array([np.linalg.norm(fn(a, tca + timedelta(seconds=x))[0] - fn(b, tca + timedelta(seconds=x))[0])
                     for x in xs])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs / 60, true, color=INK, lw=1.4, label="exact SGP4 separation")
    for step, col, ms, z in ((60, AQUA, 5, 3), (300, ORANGE, 10, 4)):
        times = frames[step]["timestamp"].drop_duplicates().sort_values()
        near = [t for t in times if abs((t - tca).total_seconds()) <= 900]
        d = [np.linalg.norm(fn(a, t)[0] - fn(b, t)[0]) for t in near]
        ax.plot([(t - tca).total_seconds() / 60 for t in near], d, "o", color=col, ms=ms, zorder=z,
                mfc="none" if step == 300 else col, mew=2,
                label=f"samples every {step} s (closest {min(d):,.0f} km)")
    ax.axhline(best["closest_approach_km"], color=BLUE, ls="--", lw=1)
    ax.annotate(f"refined miss distance {best['closest_approach_km']:.1f} km", (0, best["closest_approach_km"]),
                xytext=(10, 12), textcoords="offset points", color=BLUE)
    ax.set_yscale("log")
    ax.set_xlabel("time from closest approach (min)")
    ax.set_ylabel("separation (km)")
    ax.set_title(f"{a} – {b}: closing at {best['relative_velocity_km_s']:.1f} km/s")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "closest_approach_sampling.png", dpi=170)
    plt.close(fig)

    # Figure 2: closest approach of each pair
    top = exact.head(12)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [f"{r.satellite_a} – {r.satellite_b}" for r in top.itertuples()]
    ax.barh(labels[::-1], top["closest_approach_km"][::-1], color=BLUE, height=0.6)
    for y, v in enumerate(top["closest_approach_km"][::-1]):
        ax.text(v, y, f" {v:.0f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("closest approach in 24 h (km)")
    ax.set_title("Closest pass of the 12 nearest pairs (SGP4, refined)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "closest_approaches.png", dpi=170)
    plt.close(fig)

    # ── differential post-burn check ───────────────────────────────────
    lead = 6 * 3600
    r, v = fn(b, tca - timedelta(seconds=lead))
    nominal, _ = propagate_two_body(r, v, lead)
    rows = []
    for dv in (0.0, 0.01, 0.1, 0.5):
        burned, _ = propagate_two_body(r, apply_velocity_burn(v, dv) if dv else v, lead)
        new = np.linalg.norm(fn(b, tca)[0] + (burned - nominal) - fn(a, tca)[0])
        old = np.linalg.norm(burned - fn(a, tca)[0])
        rows.append({"delta_v_m_s": dv, "displacement_km": float(np.linalg.norm(burned - nominal)),
                     "differential_separation_km": float(new), "direct_two_body_vs_sgp4_km": float(old)})
    report["post_burn_check"] = rows
    report["two_body_vs_sgp4_6h_km"] = {
        s.name: float(np.linalg.norm(propagate_two_body(*fn(s.name, start), lead)[0] - fn(s.name, start + timedelta(seconds=lead))[0]))
        for s in sats
    }

    # ── station keeping ────────────────────────────────────────────────
    sk = simulate_tle_constellation(FLEET, start, 24, 300,
                                    station_keeping=StationKeepingConfig(decay_rate_km_per_day=1.0,
                                                                         altitude_tolerance_km=0.2))
    per = sk.groupby("satellite").agg(burns=("station_keeping_event", "sum"),
                                      delta_v_m_s=("cumulative_delta_v_m_s", "max"),
                                      fuel_percent=("fuel_percent", "min"))
    report["station_keeping_1km_per_day"] = per.reset_index().to_dict("records")
    iss = sk[sk["satellite"] == "ISS (ZARYA)"]
    hours = [(t - start).total_seconds() / 3600 for t in iss["timestamp"]]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].plot(hours, iss["controlled_mean_altitude_km"], color=BLUE, drawstyle="steps-post")
    axs[0].set_xlabel("hours"); axs[0].set_ylabel("controlled mean altitude (km)")
    axs[0].set_title("ISS: 1 km/day virtual decay, 0.2 km tolerance")
    axs[1].plot(hours, iss["fuel_remaining_kg"] * 1000, color=ORANGE, drawstyle="steps-post")
    axs[1].set_xlabel("hours"); axs[1].set_ylabel("propellant (g)")
    axs[1].set_title("Propellant (Tsiolkovsky, Isp 220 s, 12 kg)")
    fig.tight_layout()
    fig.savefig(FIG / "station_keeping.png", dpi=170)
    plt.close(fig)

    (RES / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("best_pair", "hermite_max_error_km", "post_burn_check")}, indent=1, default=str))


if __name__ == "__main__":
    main()
