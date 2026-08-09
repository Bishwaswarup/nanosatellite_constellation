# AI-Optimized Nanosatellite Constellation Manager

Phase 2 is a reproducible TLE-based constellation simulation. It propagates five sample low-Earth-orbit satellites with SGP4, transforms their positions from TEME to ECEF/ITRS, and produces ground-track and altitude telemetry. A deterministic control model applies virtual orbital decay, commands idealized altitude-restoration maneuvers, and tracks fuel use through the rocket equation.

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
- 2D ground tracks, 3D latest-position visualization, and live fuel telemetry

The controller operates on a simplified mean-orbit altitude estimate. It deliberately does not modify a TLE or alter the SGP4 ground track after a burn; physical post-maneuver propagation belongs in a later numerical-propagation phase. Conjunction screening and collision-avoidance maneuvers are the next phase.
