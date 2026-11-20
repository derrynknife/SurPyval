"""How often the joint model's link comparison picks the link the data
were simulated with, and how often it picks a richer link that adds
nothing.

The truth is ``tests/univariate/joint/_fleet.simulate_links``' fleet:
300 items followed for two years and read monthly, each marker a line
from 3.0 (sd 0.6) rising 1.0 a year (sd 0.5, correlation 0.2) read with
noise sd 0.7, and a Weibull hazard (alpha 3 years, beta 2) that follows
one link of the line: its value (0.8 per unit above 4.0), its slope (1.5
per unit a year) or the area under it (1.0 per unit-year). The effects
were chosen on a pilot of other seeds as ones the fits reach a verified
maximum for (a slope effect of 2.0 often does not). Each replicate is
fitted by ``WeibullJointPH.compare_links`` with its default five links
and ranked by AIC, 200 replicates for each link (seeds 8300 + r, 8500 + r
and 8700 + r, fixed before the first run).

- **The true link:** under each link the comparison's best is that link
  in at least 90% of the replicates (a fit that is not a verified maximum
  is not ranked, and counts against the rate).
- **A richer link that adds nothing:** under the value link, the value
  is compared with the value and the slope, and with the value and the
  area (each pair of fits from the same replicate, both verified). The
  richer is chosen only when it is more than 2 lower by AIC, twice its
  gain in log-likelihood above 4: a likelihood-ratio test at the level
  P(chi-squared on 1 df > 4) = 4.55%, the rate it must be chosen at.

Both rates are judged by ``_montecarlo.check_rate`` (3 Monte Carlo
standard errors, slack 0.01 for the second: the chi-squared reference is
asymptotic).
"""

import os
import warnings

import numpy as np
import pytest
from scipy.stats import chi2

from surpyval import WeibullJointPH
from surpyval.tests.calibration._montecarlo import check_rate, process_pool
from surpyval.tests.univariate.joint._fleet import simulate_links
from surpyval.univariate.joint import LinkComparison

UNITS = 300
REPS = 200
TRUTHS = {"value": 0.8, "slope": 1.5, "cumulative": 1.0}
FIRST = {"value": 8300, "slope": 8500, "cumulative": 8700}
#: The level of the likelihood-ratio test that choosing a link one
#: effect richer only when it is 2 lower by AIC amounts to.
LEVEL = float(chi2.sf(4.0, 1))


def replicate(truth, seed):
    data = simulate_links(
        UNITS, seed=seed, alpha=3.0, beta=2.0, **{truth: TRUTHS[truth]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comparison = WeibullJointPH.compare_links(data)
    models = comparison.models
    pairs = {}
    for richer in ("value + slope", "value + cumulative"):
        two = {k: models[k] for k in ("value", richer) if k in models}
        verified = len(two) == 2 and all(
            m.maximum == "verified" for m in two.values()
        )
        pairs[richer] = (
            LinkComparison(two, {}, "aic").best_label if verified else None
        )
    return {
        "best": comparison.best_label,
        "verified": {k: m.maximum == "verified" for k, m in models.items()},
        "pairs": pairs,
    }


def _runs(truth):
    jobs = [(truth, FIRST[truth] + r) for r in range(REPS)]
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        return list(pool.map(replicate, *zip(*jobs)))


@pytest.mark.parametrize("truth", list(TRUTHS))
def test_the_comparison_picks_the_true_link(truth):
    runs = _runs(truth)
    picks = [r["best"] for r in runs]
    shares = {
        label: picks.count(label) / REPS
        for label in sorted(set(map(str, picks)))
    }
    print(f"joint links, truth {truth}: picks {shares}")
    for label in runs[0]["verified"]:
        rate = np.mean([r["verified"].get(label, False) for r in runs])
        print(f"joint links, truth {truth}: {label} verified {rate:.3f}")
    check_rate(
        picks.count(truth),
        REPS,
        0.90,
        f"joint links: {truth} picked under {truth}",
        slack=0.0,
        side="lower",
    )
    if truth != "value":
        return
    for richer in ("value + slope", "value + cumulative"):
        chosen = [r["pairs"][richer] for r in runs]
        judged = [c for c in chosen if c is not None]
        print(f"joint links, value against {richer}: {len(judged)} judged")
        check_rate(
            judged.count(richer),
            len(judged),
            LEVEL,
            f"joint links: {richer} chosen when the extra effect is 0",
            slack=0.01,
        )
