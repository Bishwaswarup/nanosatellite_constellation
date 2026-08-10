# AI-Optimized Nanosatellite Constellation Manager

This is a reproducible, simulation-only TLE-based constellation manager. It propagates five sample low-Earth-orbit satellites with SGP4, transforms their positions from TEME to ECEF/ITRS, and produces ground-track and altitude telemetry. A deterministic controller applies virtual orbital decay, commands idealized altitude-restoration maneuvers, tracks fuel use through the rocket equation, and screens the constellation for close approaches. A lightweight AI risk score prioritizes and explains candidate events; it never replaces deterministic keep-out or post-burn clearance checks.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run main.py
```

The dashboard is available at the local Streamlit URL shown in the terminal. Use the sidebar to choose a duration and propagation timestep, then export the generated telemetry as CSV.

## Test

```bash
python3 -m pytest
```

## Scope of this milestone

- TLE ingestion for a five-satellite sample constellation
- SGP4 propagation in the TEME frame
- Astropy transformation to Earth-fixed ECEF/ITRS coordinates
- Geodetic latitude, longitude, and altitude telemetry
- Virtual orbital decay and deterministic station-keeping maneuvers
- Hohmann-transfer delta-v and Tsiolkovsky propellant accounting
- Pairwise ECEF conjunction screening and deterministic virtual avoidance planning
- RK4 two-body numerical propagation with instantaneous prograde/retrograde burn updates
- Deterministic, synthetic-data-calibrated logistic collision-risk scores using relative state, time to closest approach, predicted separation, and the keep-out distance
- AI-risk telemetry, event-log, CSV, dashboard, and synthetic-lab views
- 2D ground tracks, 3D latest-position visualization, and live fuel telemetry

The controller operates on a simplified mean-orbit altitude estimate and deliberately does not modify a TLE or alter the SGP4 ground track after a burn. Avoidance plans are re-propagated with a two-body RK4 model to the conjunction time and labelled `CLEARED` or `INSUFFICIENT`; that numerical check and the deterministic keep-out threshold remain authoritative. The AI score is simulation/decision-support only, not an operational collision-avoidance system.
