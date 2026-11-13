"""The quantiles of the log-linear regressions (PH, AFT, PO) and the
Exponential additive hazards model in closed form.

``qf`` of these families is the baseline's own inverse cumulative
hazard at a target moved by the covariates. Each family is checked on
every baseline with a closed form against two references: the
numerical inversion of the model's cumulative hazard that ``qf`` used
before (they agree to its tolerance, ``rtol=1e-12``), and the exact
quantile found to 40 digits with mpmath from the family's definition
(the closed form agrees to ``1e-12`` relative), at probabilities from
``1e-12`` to ``1 - 1e-12``. Seeds are fixed.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.regression._closed_form_qf import has_closed_form
from surpyval.univariate.regression._prediction import quantiles_by_inversion

# The clean-wheel CI job installs only the wheel and pytest.
mp = pytest.importorskip("mpmath")

P = np.array([1e-12, 1e-8, 1e-4, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-8])
P = np.append(P, 1 - 1e-12)
Z_QUERY = np.array([[0.0, 0.0], [1.0, -0.5], [0.3, 2.0]])
BASELINES: tuple[str, ...] = ("Weibull", "Exponential", "Gumbel", "Logistic")
BASELINES += ("LogLogistic",)
BASELINES += ("Normal", "LogNormal", "Gamma")
# Where each baseline's data sit: a positive or a real-line support.
POSITIVE = ("Weibull", "Exponential", "LogLogistic", "LogNormal", "Gamma")


def _data(base, seed=1):
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.binomial(1, 0.5, 120), rng.normal(0, 1, 120)])
    if base in POSITIVE:
        x = rng.weibull(1.8, 120) * 10 * np.exp(-0.3 * Z[:, 0] + 0.1 * Z[:, 1])
    else:
        x = rng.normal(10, 2, 120) - 0.8 * Z[:, 0] + 0.3 * Z[:, 1]
    c = (rng.uniform(size=120) < 0.15).astype(int)
    return x, Z, c


def _fit(name, base):
    kind = name[len(base) :]
    fitter = getattr(sp, kind)(getattr(sp, base))
    x, Z, c = _data(base)
    return fitter.fit(x=x, Z=Z, c=c)


# -- exact references --------------------------------------------------------


def _H0(base, t, a, b=None):
    if base == "Weibull":
        return (t / a) ** b
    if base == "Exponential":
        return a * t
    if base == "Gumbel":
        return mp.exp((t - a) / b)
    if base == "Logistic":
        return mp.log1p(mp.exp((t - a) / b))
    if base == "LogLogistic":
        return mp.log1p((t / a) ** b)
    if base == "Normal":
        return -mp.log(mp.ncdf(-(t - a) / b))
    if base == "LogNormal":
        return -mp.log(mp.ncdf(-(mp.log(t) - a) / b))
    if base == "Gamma":
        return -mp.log(mp.gammainc(a, b * t, mp.inf, regularized=True))
    raise ValueError(base)


def _H(kind, base, t, dist, eta):
    """The family's cumulative hazard from its definition."""
    if kind == "PH":
        return mp.exp(eta) * _H0(base, t, *dist)
    if kind == "AFT":
        return _H0(base, mp.exp(eta) * t, *dist)
    if kind == "PO":
        # S = phi S0 / (F0 + phi S0): the survival odds are phi times
        # the baseline's, so e^H - 1 = (e^H0 - 1) / phi
        return mp.log1p(mp.expm1(_H0(base, t, *dist)) / mp.exp(eta))
    if kind == "AH":
        return _H0(base, t, *dist) + t * eta
    raise ValueError(kind)


def _exact(model, kind, base, p, row, start):
    """The quantile at ``p`` and the covariate ``row`` to 40 digits."""
    params = np.asarray(model._eval_params(), dtype=float)
    k = model.k_dist
    dist = [mp.mpf(float(v)) for v in params[:k]]
    zc = np.asarray(model._centred(row[None, :]), dtype=float)[0]
    eta = mp.fsum(
        mp.mpf(float(a)) * mp.mpf(float(b)) for a, b in zip(zc, params[k:])
    )
    h = -mp.log1p(-mp.mpf(float(p)))
    with mp.workdps(60):
        return mp.findroot(
            lambda t: _H(kind, base, t, dist, eta) - h,
            mp.mpf(float(start)),
            tol=mp.mpf(10) ** -45,
        )


def _by_inversion(model, p, row):
    """The quantile by the numerical inversion ``qf`` used before."""
    params = model._eval_params()
    rows = np.asarray(model._centred(np.tile(row, (p.size, 1))), dtype=float)
    with np.errstate(all="ignore"):
        start = np.asarray(
            model.distribution.qf(p, *params[: model.k_dist]), dtype=float
        ) * np.ones(p.shape)
    return quantiles_by_inversion(
        lambda t, k: model.model.Hf(t, rows[k], *params),
        p,
        model.distribution.support,
        start,
        lambda t, k: model.model.hf(t, rows[k], *params),
    )


def _check(name, kind, base):
    model = _fit(name, base)
    assert has_closed_form(model.model)
    # A real-line quantile near 0 is held to 1e-12 of the median's size
    floor = 0.0 if base in POSITIVE else abs(float(model.qf(0.5, Z_QUERY[0])))
    for row in Z_QUERY:
        new = np.asarray(model.qf(P, row), dtype=float)
        old = _by_inversion(model, P, row)
        assert np.all(np.isfinite(new))
        # Agrees with the numerical inversion to its tolerance
        np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12)
        # and with the exact quantile to 1e-12 relative
        for p, t in zip(P, new):
            ref = float(_exact(model, kind, base, p, row, t))
            assert abs(t - ref) <= 1e-12 * max(abs(ref), floor), (
                name,
                p,
                t,
                ref,
            )


@pytest.mark.parametrize("base", BASELINES)
def test_proportional_hazards(base):
    _check(base + "PH", "PH", base)


@pytest.mark.parametrize("base", BASELINES)
def test_accelerated_failure_time(base):
    _check(base + "AFT", "AFT", base)


@pytest.mark.parametrize("base", BASELINES)
def test_proportional_odds(base):
    _check(base + "PO", "PO", base)


def test_exponential_additive_hazards():
    _check("ExponentialAH", "AH", "Exponential")


def test_no_closed_form_keeps_the_inversion():
    """A custom link, another additive baseline and an accelerated life
    model have no closed form; their quantiles are as before."""
    x, Z, c = _data("Weibull")
    models = [
        sp.WeibullAH.fit(x=x, Z=Z, c=c),
        sp.AcceleratedLife(sp.Weibull, sp.life_models.GeneralLogLinear).fit(
            x=x, Z=Z, c=c
        ),
    ]
    for model in models:
        assert not has_closed_form(model.model)
        for row in Z_QUERY[:2]:
            np.testing.assert_array_equal(
                model.qf(P[2:-2], row), _by_inversion(model, P[2:-2], row)
            )


def test_ends_and_missing():
    """``qf(0)`` and ``qf(1)`` are the ends of the support, nan is nan,
    as before."""
    for base, kind in (
        ("Weibull", "PH"),
        ("Normal", "AFT"),
        ("Logistic", "PO"),
    ):
        model = _fit(base + kind, base)
        lower, upper = model.distribution.support
        out = model.qf([0.0, 1.0, np.nan], Z_QUERY[1])
        assert out[0] == lower and out[1] == upper and np.isnan(out[2])


def test_extreme_covariates():
    """A linear predictor that sends the target beyond the floats is
    solved by the inversion, as before, rather than returned as nan."""
    model = _fit("WeibullPH", "Weibull")
    row = np.array([0.0, 1e4])
    with np.errstate(all="ignore"):
        new = model.qf(P[2:-2], row)
        old = _by_inversion(model, P[2:-2], row)
    np.testing.assert_array_equal(np.isfinite(new), np.isfinite(old))
    ok = np.isfinite(new) & (old > 0)
    np.testing.assert_allclose(new[ok], old[ok], rtol=1e-12)
