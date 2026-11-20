# Coverage of `ParametricMultiState` occupancy bounds

Does `ParametricMultiState.cb` (the delta-method bound on a state occupation probability) cover the truth at its stated level, in particular for a state that is rarely occupied? Compared with a parametric-bootstrap band on the same data.

Everything in "Design" was written on 20 November 2026, before any replicate with a study seed was run. Results are below it.

## Design

**Models** (all `clock="forward"`; states healthy, ill, dead; every unit healthy at 0). A transition's hazard is Weibull in the time since the start, `H(t) = (t / scale) ** shape`.

| Design | Fitted `dist` | healthy → ill | healthy → dead | ill → dead |
|---|---|---|---|---|
| `id_exp_rare` | Exponential | rate 0.02 | rate 0.10 | rate 0.5 |
| `id_exp_common` | Exponential | rate 0.15 | rate 0.05 | rate 0.10 |
| `prog_exp_rare` | Exponential | rate 0.03 | none | rate 0.6 |
| `prog_exp_common` | Exponential | rate 0.2 | none | rate 0.15 |
| `id_wei_rare` | Weibull | (35, 1.3) | (10, 1.5) | (2, 1.2) |
| `id_wei_common` | Weibull | (6, 1.3) | (20, 1.5) | (12, 1.2) |
| `prog_wei_rare` | Weibull | (30, 1.2) | none | (2, 1.3) |
| `prog_wei_common` | Weibull | (6, 1.3) | none | (10, 1.2) |

`id_*` is an illness-death model, `prog_*` a progressive one (healthy → ill → dead). In the `rare` designs the probability of being ill is between 1% and 5% at each query time; in the `common` ones it is 20% to 47%. The true values are printed by `study.py truth`.

**Data.** `n` = 100, 300, 1000 units, each followed from 0 to its own uniform (10, 20) end. 1000 replicates of each design and size, seeds 1000 to 1999: replicate `seed` of design number `d` (in the table's order, from 0) at size `n` uses `numpy.random.default_rng([d, n, seed])`.

**Targets** (13 per design): the probability of being healthy, ill and dead at 2, 5 and 10 (from the start), and of being ill, and dead, at 3 and 5 for a unit that is ill at 2 (`from_state="ill"`, `s=2`). The last four depend on the ill → dead transition alone, which in a rare design is fitted from few spells.

**Truth.** The formula in `study.py` (`formula`, 2000 Gauss-Legendre nodes): exact for the survival terms, a one-dimensional integral for the probability of being ill. `study.py truth` checks it against `scipy.integrate.quad` and against 10^6 simulated paths from the study's own data generator. (A formula rather than the simulation as the truth: the simulation's own error, about 1.4e-4 on a probability of 0.02, would be a visible share of a band's half-width at n = 1000.)

**Methods**, two-sided at 90% and 95%:

- `exp`: `cb(bound_type="exp")`, the default, on the log(-log) scale.
- `normal`: `cb(bound_type="normal")`.
- `logit`: the same estimate and standard error carried to the logit scale (computed in the study; a candidate).
- `boot`: a parametric bootstrap by parameter draws, B = 500: every transition's parameters drawn from the normal approximation to their estimates on the log scale (the fitted covariance divided by the outer product of the estimates), transitions independent, the occupation probability computed at each draw by `formula` (200 nodes), and the band the draws' percentiles. This is the simulation interval of R's flexsurv. The draws use `default_rng([d, n, seed, 1])`. It is not a refit of data simulated from the fit: 500 refits for each of 24,000 replicates was out of reach.

`exp` and `normal` are `surpyval.multistate._base.probability_bounds` of the model's `occupancy` and `occupancy_se`, which is what `cb` computes; each replicate also calls `cb` itself for two targets and requires the same numbers. Each replicate records the largest gap between `formula` at the fitted parameters and the model's `occupancy`.

**Recorded** per cell (design, size, target, method, level): coverage, the two one-sided miss rates (the whole band below the truth, the whole band above it), the mean width.

**Rules, fixed beforehand.**

- A replicate whose fit raises (a transition with no events, say) is counted and reported per cell, and left out of that cell's coverage. A band with a missing (`nan`) end is a miss.
- A cell under-covers when its coverage is below nominal by more than 3 Monte Carlo standard errors (1000 replicates: below 0.9293 at 95%, below 0.8715 at 90%). With 8 × 3 × 13 × 2 = 624 cells per method, about one cell is expected below that line by chance for a method that is exactly right; the reading is of the pattern (which states, which sizes), not of one cell.
- Nothing above changes after the first run. A marginal result is reported as it is.

## Results (run 20 November 2026)

24,000 replicates; every fit was made. The largest gap between `formula` and the model's `occupancy` was 1.1e-8. Every cell is in `table.txt` (and `summary.json`); the raw replicates are not kept.

**One correction to the harness after the first run, none to the design.** The per-replicate check that `cb` gives the numbers of `probability_bounds` asked for exact equality; `cb` at one time and at several can differ in the last digit (2.8e-17 seen), so 89 replicates of the two exponential progressive designs stopped on that check and were first recorded as failed fits. The check now allows 1e-12 and those 89 replicates (same seeds) were run again. No other replicate was re-run, and no design, seed, size, target or threshold changed.

**Mean coverage over a size's 104 cells:**

| Method | 90%: n = 100 | 300 | 1000 | 95%: n = 100 | 300 | 1000 |
|---|---|---|---|---|---|---|
| `exp` (the default) | 0.893 | 0.901 | 0.902 | 0.945 | 0.951 | 0.950 |
| `normal` | 0.879 | 0.895 | 0.901 | 0.929 | 0.945 | 0.948 |
| `logit` | 0.895 | 0.902 | 0.902 | 0.947 | 0.952 | 0.950 |
| `boot` | 0.889 | 0.898 | 0.899 | 0.939 | 0.946 | 0.946 |

**Cells below nominal by more than 3 Monte Carlo standard errors** (of 104 per size and level):

| Method | 90%: n = 100 | 300 | 1000 | 95%: n = 100 | 300 | 1000 |
|---|---|---|---|---|---|---|
| `exp` | 4 | 0 | 0 | 9 | 0 | 0 |
| `normal` | 28 | 2 | 0 | 32 | 10 | 0 |
| `logit` | 3 | 0 | 0 | 6 | 0 | 0 |
| `boot` | 14 | 0 | 0 | 18 | 2 | 4 |

**What it says.**

- **At 300 and 1000 units the default bound holds its level for every state, rare or common:** no cell of 416 is below the line. The lowest are 0.933 (95%) and 0.878 (90%).
- **At 100 units it is two to four points short where a rare state is involved,** and only there: all 13 of its flagged cells are in the `rare` designs at n = 100, 10 of them the probability of being ill (truth 1% to 4%). The lowest are 0.912 at 95% and 0.860 at 90% (`id_wei_rare`, ill at 10). In these cells the ill → dead hazard is fitted from roughly 10 to 35 spells. The misses are lopsided: the whole band lies below the truth about twice to three times as often as above it (0.060 against 0.017 at 95%, averaged over the rare Weibull designs' ill cells), and the reported standard error is mostly 3% to 7% smaller than the estimate's actual spread (its ratio to the spread is 0.93 to 1.02 in those cells, and 0.86 to 0.90 for a unit already ill). That is the ordinary small-sample behaviour of a Wald bound on few events, not an error in the delta method: it goes with the number of transitions observed, and is gone by 300 units.
- **`normal` is clearly worse** for rare states (down to 0.850 at 95%); it is not the default.
- **`logit` is about half a point to a point better than `exp` in the short cells and the same elsewhere,** and is still short in most of them (6 flagged cells against 9 at 95%, lowest 0.920). It does not fix the shortfall.
- **The parameter-draw bootstrap is not better.** In the rare Weibull designs at n = 100 and 95%, it does better on the probability of being ill from the start in five cells of six (0.952 to 0.964 where `exp` has 0.912 to 0.942; the sixth is 0.906) but worse for a unit already ill (0.849 to 0.900 where `exp` has 0.926 to 0.941), and it stays short of 95% at n = 1000 in four cells.

**Width.** In the common designs and the exponential ones the default band and the bootstrap band have about the same mean width (ratio 0.90 to 1.16 at n = 100, 0.97 to 1.07 at 300, 1.00 to 1.03 at 1000). In the rare Weibull designs the default band is the narrower: on the ill and from-ill targets the ratio of mean widths (95%) is 0.37 to 0.97 at n = 100, 0.63 to 1.00 at 300 and 0.87 to 1.01 at 1000. A ratio in that range is therefore what the two methods give on a rare state, and in those same cells the wider band covers no better (worse, for a unit already ill). A narrower delta-method band than a simulated one is not, by itself, a sign of under-coverage.

**Outcome.** The default stays `bound_type="exp"`; no number the package computes changes. Neither candidate is a fix for the small-sample shortfall, so none was added. A bound that should do better with few transitions (a profile-likelihood bound on the occupation probability, or a bootstrap that refits) is not built; the documentation now says what to expect with few transitions. `surpyval/tests/calibration/test_coverage_multistate_parametric.py` keeps a cut-down version (two rare designs and the regression model at one covariate value, 300 units, 400 replicates) in the calibration suite.
