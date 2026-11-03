"""Draws from a regression model follow its survival curve (#285).

For every registered regression with ``random``, 20,000 draws at one
covariate row (with the case's ``call_kwargs``, such as a stratum) are
compared with the model's ``sf`` at that row by the one-sample
Kolmogorov-Smirnov distance, which must be inside the band that a correct
sampler leaves only 0.1% of the time. The distance is taken over every
time, including the gaps between the draws, where a step model's curve
jumps with no draw on its step, and beyond the last finite draw (a draw
of ``inf`` is a unit that outlives a step estimate). The draws are
seeded, so the test is deterministic. A regression with no ``random``
excludes the property with its reason.
"""

import numpy as np
import pytest

from surpyval.tests._helpers import KS_BAND_20000, ks_distance
from surpyval.tests.conformance.registry import call, cases_for, fitted

N_DRAWS = 20_000
SEED = 20261103
BAND = KS_BAND_20000


@pytest.mark.parametrize("case", cases_for("random_sf"))
def test_random_follows_sf(case):
    model = fitted(case)
    assert hasattr(model, "random"), (
        f"{case.name} has no random(): add one, or exclude 'random_sf' "
        "with the reason"
    )
    # The first query row (a single stress for a one-stress model).
    row = np.asarray(case.Z, dtype=float)[:1]
    kwargs = dict(case.call_kwargs)
    draws, rows = model.random(N_DRAWS, row, random_state=SEED, **kwargs)
    draws = np.asarray(draws, dtype=float)
    assert draws.shape == (N_DRAWS,)
    assert np.asarray(rows).shape[0] == N_DRAWS
    assert not np.isnan(draws).any(), "nan draws at an ordinary row"

    def ff(x):
        return call(case, model, "ff", np.asarray(x, dtype=float), row)

    d = ks_distance(draws, ff)
    assert d <= BAND, (
        f"{case.name}: KS distance {d:.4f} of {N_DRAWS} draws from ff "
        f"is beyond the 0.1% band {BAND:.4f}"
    )
