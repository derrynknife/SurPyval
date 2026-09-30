"""AFT and PO fits take the gradient ladder first (#499).

``optimise_nm_tnc`` ran Nelder-Mead before anything else, which was 85% of
an AFT or PO fit. With a differentiable objective it now tries
``optimise_ph`` first and keeps that answer when it is a verified optimum.
These tests pin that the answer is at least as good as the old ladder's,
and that Nelder-Mead is not called when the gradient ladder converges.
"""

from unittest import mock

import numpy as np
import pytest
from scipy.optimize import minimize

import surpyval as sp
from surpyval.univariate.regression import _fit_skeleton
from surpyval.univariate.regression.accelerated_failure_time import (
    aft_fitter,
)
from surpyval.univariate.regression.proportional_odds import (
    proportional_odds_fitter,
)

FITTERS = [
    (sp.WeibullAFT, aft_fitter),
    (sp.LogNormalAFT, aft_fitter),
    (sp.WeibullPO, proportional_odds_fitter),
]


def _data(n=400, seed=499):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    x = rng.weibull(1.5, n) * 50 * np.exp(Z @ np.array([0.3, -0.2, 0.1]))
    c = (rng.random(n) < 0.3).astype(int)
    return x, Z, c


def _capture_objective(fitter, module):
    """The objective and start the fitter hands to ``optimise_nm_tnc``."""
    captured = []
    real = _fit_skeleton.optimise_nm_tnc

    def spy(fun, init_t, quiet=False):
        captured.append((fun, np.array(init_t, dtype=float)))
        return real(fun, init_t, quiet=quiet)

    with mock.patch.object(module, "optimise_nm_tnc", spy):
        fitter.fit(*_data())
    assert captured, "the fitter did not call optimise_nm_tnc"
    return captured[0]


def _legacy_ladder(fun, init_t):
    res = minimize(
        fun, init_t, method="Nelder-Mead", options={"maxiter": 1000}
    )
    res2 = minimize(fun, res.x, method="TNC")
    return res2 if res2.success else res


@pytest.mark.parametrize("fitter,module", FITTERS)
def test_reaches_at_least_the_old_ladders_optimum(fitter, module):
    fun, init_t = _capture_objective(fitter, module)
    new = _fit_skeleton.optimise_nm_tnc(fun, init_t, quiet=True)
    old = _legacy_ladder(fun, init_t)
    assert new.fun <= old.fun + 1e-6 * max(1.0, abs(old.fun))


@pytest.mark.parametrize("fitter,module", FITTERS)
def test_no_nelder_mead_when_the_gradient_ladder_converges(fitter, module):
    fun, init_t = _capture_objective(fitter, module)
    methods = []
    real_minimize = _fit_skeleton.minimize

    def recording(*args, **kwargs):
        methods.append(kwargs.get("method"))
        return real_minimize(*args, **kwargs)

    with mock.patch.object(_fit_skeleton, "minimize", recording):
        res = _fit_skeleton.optimise_nm_tnc(fun, init_t, quiet=True)
    assert np.isfinite(res.fun)
    assert not getattr(res, "stopped_short", False)
    assert "Nelder-Mead" not in methods, methods
