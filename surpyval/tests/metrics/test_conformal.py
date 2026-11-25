"""Conformal lower bounds on a survival time (``conformal_lower_bound``)."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.metrics import ConformalLowerBound, conformal_lower_bound

BETA = np.array([0.8, -0.5])
SHAPE, SCALE = 1.5, 10.0


def _draw(rng, n):
    """Weibull proportional hazards in two normal covariates."""
    Z = rng.normal(size=(n, 2))
    T = SCALE * (rng.exponential(size=n) / np.exp(Z @ BETA)) ** (1 / SHAPE)
    return Z, T


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(723)
    Z, T = _draw(rng, 300)
    cz = rng.uniform(0, 30, 300)
    x, c = np.minimum(T, cz), (cz < T).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "right": (sp.WeibullPH.fit(x=x, Z=Z, c=c), True),
            "wrong family": (sp.ExponentialPH.fit(x=x, Z=Z, c=c), True),
            "no covariates": (sp.Weibull.fit(x, c), False),
        }


def _coverage(model, use_Z, cmax, method, reps=200, n=200, alpha=0.1):
    covs = []
    for r in range(reps):
        g = np.random.default_rng([723, r])
        Zc, Tc = _draw(g, n)
        cz = cmax * g.uniform(size=n) if np.isfinite(cmax) else np.inf
        x, c = np.minimum(Tc, cz), (cz < Tc).astype(int)
        b = conformal_lower_bound(
            model, x, c, Z=Zc if use_Z else None, alpha=alpha, method=method
        )
        Zt, Tt = _draw(g, 2000)
        L = b.lower(Zt) if use_Z else b.lower()
        covs.append(np.mean(Tt >= L))
    covs = np.asarray(covs)
    return covs.mean(), covs.std() / np.sqrt(reps)


@pytest.mark.parametrize("which", ["right", "wrong family", "no covariates"])
def test_nominal_coverage_without_censoring(models, which):
    # Plain split conformal: coverage between 1 - alpha and 1 - alpha +
    # 1/(n + 1) = 0.905 on average, whatever the model.
    model, use_Z = models[which]
    cov, se = _coverage(model, use_Z, np.inf, "observed")
    assert 0.9 - 3 * se <= cov <= 0.905 + 3 * se, (cov, se)


@pytest.mark.parametrize("which", ["right", "wrong family", "no covariates"])
def test_observed_method_covers_with_censoring(models, which):
    # About 34% and 71% censored: never below 90%, conservative above it
    # (0.95 and 0.99 for the right model).
    model, use_Z = models[which]
    for cmax, at_least in ((30.0, 0.93), (8.0, 0.97)):
        cov, se = _coverage(model, use_Z, cmax, "observed")
        assert cov >= at_least, (which, cmax, cov)


@pytest.mark.parametrize("which", ["right", "wrong family", "no covariates"])
def test_ipcw_method_near_nominal_with_censoring(models, which):
    # 34% censored: 0.909 for each model (Monte Carlo se 0.0015);
    # 71% censored: about 0.945, still above the level.
    model, use_Z = models[which]
    cov, se = _coverage(model, use_Z, 30.0, "ipcw")
    assert 0.9 - 3 * se <= cov <= 0.92, (which, cov, se)
    cov, se = _coverage(model, use_Z, 8.0, "ipcw")
    assert 0.9 - 3 * se <= cov <= 0.96, (which, cov, se)


def test_reduces_to_plain_split_conformal():
    # No censoring: the level is the floor(alpha (n + 1))-th smallest
    # predicted CDF at the calibration times.
    rng = np.random.default_rng(5)
    x = 10 * rng.weibull(2.0, 99)
    model = sp.Weibull.from_params([12.0, 1.5])
    for alpha in (0.05, 0.1, 0.2):
        b = conformal_lower_bound(model, x, alpha=alpha)
        u = np.sort(model.ff(x))
        k = int(np.floor(alpha * 100))
        assert b.level == u[k - 1]
        assert b.lower()[0] == pytest.approx(model.qf(u[k - 1]), rel=1e-12)
        # ipcw with no censoring is the same.
        assert conformal_lower_bound(
            model, x, alpha=alpha, method="ipcw"
        ).level == pytest.approx(b.level, rel=1e-15)


def test_observed_censoring_by_hand():
    # Five units, alpha = 1/3: floor(6/3) = 2, the second smallest of the
    # observed CDF values, censored ones at their censoring time.
    model = sp.Exponential.from_params([0.1])
    x = np.array([1.0, 3.0, 2.0, 8.0, 20.0])
    c = np.array([0, 1, 1, 0, 0])
    b = conformal_lower_bound(model, x, c, alpha=1 / 3)
    assert b.level == pytest.approx(1 - np.exp(-0.2))
    assert b.lower()[0] == pytest.approx(2.0)
    assert b.n_censored == 2 and b.n_calibration == 5


def test_ipcw_by_hand():
    # Events at 1, 8, 20; censored at 2, 3. Reverse Kaplan-Meier (events
    # first): G(1-) = 1, G(8-) = G(20-) = 1 * 3/4 * 2/3 = 1/2. Weights 1,
    # 2, 2, scaled to sum 5/6: 1/6, 1/3, 1/3. Mass below the second score
    # is 1/6, plus 1/6 for the new unit: 1/3 <= alpha = 0.34, so the level
    # is F(8).
    model = sp.Exponential.from_params([0.1])
    x = np.array([1.0, 3.0, 2.0, 8.0, 20.0])
    c = np.array([0, 1, 1, 0, 0])
    b = conformal_lower_bound(model, x, c, alpha=0.34, method="ipcw")
    assert b.level == pytest.approx(1 - np.exp(-0.8))
    b = conformal_lower_bound(model, x, c, alpha=0.32, method="ipcw")
    assert b.level == pytest.approx(1 - np.exp(-0.1))


def test_counts_expand_rows():
    model = sp.Weibull.from_params([10.0, 2.0])
    x = np.array([2.0, 5.0, 9.0, 12.0])
    c = np.array([0, 1, 0, 0])
    n = np.array([3, 2, 4, 1])
    a = conformal_lower_bound(model, x, c, n, alpha=0.2)
    b = conformal_lower_bound(
        model, np.repeat(x, n), np.repeat(c, n), alpha=0.2
    )
    assert a.level == b.level and a.n_calibration == 10


def test_model_without_qf_is_inverted_from_its_cdf():
    # A forest has no qf: its bound is found from sf by bisection, and
    # equals the generalised inverse of its step-function CDF.
    from surpyval.ml import RandomSurvivalForest

    rng = np.random.default_rng(2)
    Z, T = _draw(rng, 300)
    forest = RandomSurvivalForest.fit(x=T[:200], Z=Z[:200], n_trees=10)
    b = conformal_lower_bound(forest, T[200:], Z=Z[200:], alpha=0.1)
    Znew = Z[:5]
    L = b.lower(Znew)
    assert np.all(np.isfinite(L)) and np.all(L > 0)
    F = 1 - forest.sf(L, Znew)
    F_below = 1 - forest.sf(L * (1 - 1e-9), Znew)
    assert np.all(F >= b.level) and np.all(F_below < b.level)


def test_bisection_matches_qf():
    from surpyval.metrics.conformal import _invert

    model = sp.Weibull.from_params([10.0, 2.0])
    q = _invert(lambda t: model.ff(t), 0.1, 1)
    assert q[0] == pytest.approx(model.qf(0.1), rel=1e-12)
    # A CDF that never reaches the level: inf.
    assert np.isinf(_invert(lambda t: np.full_like(t, 0.05), 0.1, 2)).all()


def test_too_few_units_warns_and_bounds_zero():
    model = sp.Weibull.from_params([10.0, 2.0])
    with pytest.warns(UserWarning, match="at least 9 are needed"):
        b = conformal_lower_bound(model, [1.0, 2.0, 3.0], alpha=0.1)
    assert b.level == 0 and b.lower()[0] == 0


def test_repr_and_type():
    b = conformal_lower_bound(
        sp.Weibull.from_params([10.0, 2.0]), np.arange(1.0, 30.0)
    )
    assert isinstance(b, ConformalLowerBound)
    assert "90% of new units outlive it" in repr(b)
    assert sp.conformal_lower_bound is conformal_lower_bound


@pytest.mark.parametrize(
    "kwargs, words",
    [
        ({"alpha": 0}, "strictly between 0 and 1"),
        ({"method": "cqr"}, "'method' must be one of"),
        ({"c": [0, -1, 0]}, "left and"),
        ({"c": [0, 1]}, "one flag per row"),
        ({"n": [1, 0.5, 1]}, "whole"),
        ({"Z": [[1.0], [2.0]]}, "rows for 3 rows"),
    ],
)
def test_refusals(kwargs, words):
    with pytest.raises(ValueError, match=words):
        conformal_lower_bound(
            sp.Weibull.from_params([10.0, 2.0]), [1.0, 2.0, 3.0], **kwargs
        )


def test_cox_past_its_last_failure_is_inf_not_nan():
    # Cox's step baseline ends at the last failure: a low-risk unit's CDF
    # never reaches the level, and its bound is inf (qf alone gives nan).
    rng = np.random.default_rng(9)
    Z, T = _draw(rng, 300)
    x, c = np.minimum(T, 6.0), (T > 6.0).astype(int)
    cox = sp.CoxPH.fit(x=x[:200], Z=Z[:200], c=c[:200])
    b = conformal_lower_bound(cox, x[200:], c[200:], Z=Z[200:], alpha=0.3)
    L = b.lower(np.array([[-3.0, 3.0], [0.0, 0.0]]))
    assert np.isinf(L[0]) and np.isfinite(L[1]) and L[1] > 0
