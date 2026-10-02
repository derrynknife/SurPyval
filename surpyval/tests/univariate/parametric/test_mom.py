"""The method of moments: fixed parameters, the search, starts that are
never NaN, and the numeric path with an offset.
"""

import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval import Weibull


def test_mom_never_returns_a_nan_objective_start():
    np.random.seed(0)
    x = surv.BetaGeometric.random(500, 5.0, 3.0)
    # the default start a = 1 has no finite mean
    with pytest.raises(ValueError, match="moments are not finite"):
        surv.BetaGeometric.fit(x, how="MOM", fixed={"b": 3.0})
    # a valid start works
    model = surv.BetaGeometric.fit(x, how="MOM", fixed={"b": 3.0}, init=[4.0])
    assert model.moment(1) == pytest.approx(x.mean(), rel=1e-4)


def test_mom_with_fixed_matches_only_the_free_moments():
    np.random.seed(1)
    x = surv.Weibull.random(50, 10, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = surv.Weibull.fit(x, how="MOM", fixed={"beta": 2})
    assert model.params[1] == 2
    # one free parameter, one equation: the mean is matched exactly
    assert model.mean() == pytest.approx(x.mean(), rel=1e-4)


def test_mom_search_backs_away_from_missing_moments():
    # With alpha fixed well below the data's scale, matching the mean
    # needs a LogLogistic shape just above 1, next to the region where the
    # mean does not exist. The search used to step in, end on a nan
    # objective and report it; that region now reads as +inf.
    np.random.seed(0)
    x = surv.LogLogistic.random(100, 10, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = surv.LogLogistic.fit(x, how="MOM", fixed={"alpha": 2.0})
    assert 1 < model.params[1] < 1.5
    assert model.mean() == pytest.approx(x.mean(), rel=1e-4)


def test_mom_with_every_parameter_fixed():
    model = surv.Weibull.fit(
        [1.0, 2.0, 3.0], how="MOM", fixed={"alpha": 3.0, "beta": 2.0}
    )
    np.testing.assert_allclose(model.params, [3.0, 2.0])


# ---------------------------------------------------------------------------
# #275: the numeric MOM path optimises to convergence.
# ---------------------------------------------------------------------------


class TestMOMNumericPath:
    def test_mom_offset_recovers_parameters(self):
        # 275: tol=1e-1 used to stop the optimiser at beta ~ 3-4.
        np.random.seed(12)
        T = 50 + 10 * np.random.weibull(2, 5000)
        m = Weibull.fit(x=T, how="MOM", offset=True)
        assert m.gamma == pytest.approx(50.0, abs=1.0)
        assert m.params[0] == pytest.approx(10.0, rel=0.1)
        assert m.params[1] == pytest.approx(2.0, rel=0.1)
