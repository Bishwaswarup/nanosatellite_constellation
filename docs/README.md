# Physics guide

`physics_guide.tex` is a self-contained LaTeX technical companion to the simulation. Its diagrams and graphs are vector drawings made with TikZ and PGFPlots, so no separate image assets are required.

Compile it with a TeX distribution that includes TikZ and PGFPlots:

```bash
pdflatex -output-directory build docs/physics_guide.tex
pdflatex -output-directory build docs/physics_guide.tex
```

Running LaTeX twice resolves the table of contents and references. The resulting PDF is `build/physics_guide.pdf`.
