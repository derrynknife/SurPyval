"""The interval-censored Cox model (``surpyval.IntervalCoxPH``) against
stored R ``icenReg::ic_sp(model = "ph")`` fits
(``scripts/reference/reference_r_interval_cox.R``).

``ic_sp`` maximises the same likelihood over the same Turnbull intervals
(called with ``B = c(0, 1)``, SurPyval's ``(l, r]`` convention), by
iterative convex minorant steps on the baseline and Newton steps on the
coefficients, so the fits agree to the solvers' tolerances. The
tolerances were fixed before the comparison: coefficients and
log-likelihoods to 1e-4 (absolute), and the predicted survival after
each Turnbull interval at two covariate rows to 1e-4.

Standard errors: ``ic_sp`` gives only the bootstrap's. SurPyval's case
bootstrap of the same size (1000 refits) is a different random draw, so
its standard error is compared in distribution: with 1000 refits each
estimate of a standard error has a Monte Carlo error of about 2.2%, and
the two are required to agree within 15% (fixed before either was run).
The default, profile-likelihood standard error estimates the same
quantity by another route and is held to the same 15% of R's bootstrap
(a sanity bound; values on icenReg's own data, within 6%, were seen
while choosing the method).

The data are seeded synthetic fixtures from ``make_fixtures.py``: current
status data of the kind of icenReg's miceData, interval data with wide
and narrow intervals, exact times and shared whole-year endpoints of the
kind of its IR_diabetes, and a mixed simulated set. icenReg's own data
are not copied into the repository; the same checks on them were run
locally (see DECISIONS.md).

With exact and right-censored data only (``exact_right``) the model is
the grouped-data proportional hazards model: ``ic_sp`` and SurPyval
agree, and both differ from ``coxph`` (whose partial likelihood treats
the baseline as continuous) by a little, recorded here.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, reference, values

SOURCE = "r_interval_cox"


def _rows(l, u):
    """SurPyval's ``x`` and ``c`` from icenReg's ``[l, u]`` columns
    (``u`` NaN is Inf; ``l == 0`` before the first visit is left
    censored)."""
    u = np.where(np.isnan(u), np.inf, u)
    c = np.where(l == u, 0, np.where(np.isinf(u), 1, np.where(l <= 0, -1, 2)))
    x = np.column_stack([np.where(c == -1, u, l), np.where(c == 1, l, u)])
    return x, c


def _data(ref_id):
    if ref_id == "current_status":
        d = fixture("ic_current_status")
        x, c = _rows(d["l"], d["u"])
        return x, c, d["grp"].reshape(-1, 1)
    if ref_id == "wide_narrow":
        d = fixture("ic_wide_narrow")
        x, c = _rows(d["left"], d["right"])
        return x, c, d["male"].reshape(-1, 1)
    d = fixture("ic_mixed")
    Z = np.column_stack([d["z1"], d["z2"]])
    if ref_id == "mixed":
        x, c = _rows(d["l"], d["u"])
        return x, c, Z
    c = 1 - d["er_event"].astype(int)
    return d["er_time"], c, Z


IDS = ("current_status", "wide_narrow", "mixed", "exact_right")


@pytest.fixture(scope="module")
def fits():
    out = {}
    for ref_id in IDS:
        x, c, Z = _data(ref_id)
        out[ref_id] = sp.IntervalCoxPH.fit(x, Z, c=c)
    return out


@pytest.mark.parametrize("ref_id", IDS)
def test_coefficients_and_log_likelihood(fits, ref_id):
    r = values(SOURCE, ref_id)
    model = fits[ref_id]
    assert model.maximum == "verified"
    assert_allclose(model.beta, r["coef"], atol=1e-4)
    assert_allclose(model.log_likelihood, r["loglik"], atol=1e-4)


@pytest.mark.parametrize("ref_id", IDS)
def test_survival_after_each_turnbull_interval(fits, ref_id):
    r = values(SOURCE, ref_id)
    model = fits[ref_id]
    newdata = reference(SOURCE, ref_id)["values"]["newdata"]
    rows = np.column_stack([np.asarray(v, float) for v in newdata.values()])
    # icenReg's first row is the start (survival 1 before the first
    # interval), then one row per Turnbull interval
    upper = r["upper"][1:]
    finite = np.isfinite(upper)
    for k, key in enumerate(("sf_row1", "sf_row2")):
        expected = r[key][1:][finite]
        got = model.sf(
            upper[finite], np.repeat(rows[k : k + 1], finite.sum(), 0)
        )
        assert_allclose(got, expected, atol=1e-4)
    # The same Turnbull intervals
    assert_allclose(model.intervals[:, 1][finite], upper[finite])


# (exact_right's, a minute of refits, is left to the interval data)
@pytest.mark.parametrize("ref_id", ("current_status", "wide_narrow", "mixed"))
def test_standard_errors_against_the_bootstrap(fits, ref_id):
    r = values(SOURCE, ref_id)
    model = fits[ref_id]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = model.standard_errors(
            "bootstrap", n_boot=1000, random_state=20261113
        )
    assert_allclose(boot, r["bootstrap_se"], rtol=0.15)
    assert_allclose(model.standard_errors(), r["bootstrap_se"], rtol=0.15)


def test_exact_and_right_censored_data_differ_from_coxph_a_little(fits):
    """The grouped-data likelihood (``ic_sp`` and SurPyval) against Cox's
    partial likelihood: close, not equal."""
    ours = fits["exact_right"].beta
    for ties in ("breslow", "efron"):
        cox = values(SOURCE, f"exact_right_coxph_{ties}")["coef"]
        assert np.all(np.abs(ours - cox) < 0.1 * np.abs(cox))
        assert np.max(np.abs(ours - cox)) > 1e-4
