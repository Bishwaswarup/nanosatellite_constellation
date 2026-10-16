# Physics guide

[`physics_guide.pdf`](physics_guide.pdf) is a self-contained technical companion to the simulation:
SGP4 and frame conversion, the station-keeping loop and rocket equation, between-sample conjunction
screening, the differential post-burn check and the advisory risk score. Its diagrams are TikZ and
PGFPlots, so the source [`physics_guide.tex`](physics_guide.tex) needs no image files.

Rebuild it with a TeX distribution that includes TikZ and PGFPlots:

```bash
latexmk -pdf -outdir=build docs/physics_guide.tex
```
