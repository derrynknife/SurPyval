# Joint model recovery study (#891)

200 simulated fleets per path model (300 pumps each, read monthly for up to two years; Weibull PH hazard with alpha 800 days and beta 2 at a marker of 4; marker effect 0.9), fitted with `marker_reference=4`. `docs/Joint Models.rst` quotes the results ("How well it recovers the truth"), dated 2026-10-16. The page does not re-run the study.

- `study.py <path> [reps]`: runs one path model (`linear`, `exp`, `quad`, `loglink`) and writes `results_<path>.jsonl` and `summary_<path>.json` here.
  - It resumes a killed run and uses at most 4 processes.
  - `GH=<points>` refits with that many quadrature points per dimension; `N=<pumps>` sets the fleet size. These are the diagnostic runs, tagged `_gh11` and `_n1200`.
- `table.py`: prints the per-parameter table (`table.txt`) from the summaries.
- `beta_bias.py`: the small-sample bias of a plain Weibull shape estimate at the study's size, for comparison.

Run from the repository root with the package importable (e.g. `pip install -e .`). The raw `results_*.jsonl` files are not kept; the summaries are.
