"""
Parametric fits are equivariant under a change of the data's units.

Fitting ``k * x`` must give the fit to ``x`` with every scale parameter
multiplied by ``k`` (rates divided by it, a log-location shifted by
``log k``) and every shape parameter unchanged -- for small ``k`` as well
as large. Before round 4 the optimiser's rescaling only scaled *up*: a
parameter starting below 1 kept a unit step, so at data scales of 1e-3 a
Rayleigh MOM fit was 1.2% off, a Uniform MPS fit 0.15% and a Beta4 MLE
fit 0.1%.
"""

import warnings
from typing import Any

import numpy as np
import pytest

import surpyval as surv
from surpyval.univariate.parametric.fitters import search_floor

SCALES = (1e-4, 1e-3, 1e3, 1e5)
TOL = 1e-4

# Generating parameters, and how each parameter maps when x -> k x:
# "s" scale or location (times k), "r" rate (divided by k), "l"
# log-location (plus log k), "i" shape (unchanged).
FAMILIES: dict[str, tuple[tuple[float, ...], str]] = {
    "Weibull": ((10.0, 2.0), "si"),
    "Exponential": ((0.1,), "r"),
    "Gamma": ((2.0, 0.5), "ir"),
    "Normal": ((5.0, 2.0), "ss"),
    "LogNormal": ((1.0, 0.5), "li"),
    "Logistic": ((5.0, 2.0), "ss"),
    "LogLogistic": ((10.0, 3.0), "si"),
    "Gumbel": ((5.0, 2.0), "ss"),
    "GumbelLEV": ((5.0, 2.0), "ss"),
    "Rayleigh": ((3.0,), "s"),
    "ExpoWeibull": ((10.0, 2.0, 1.5), "sii"),
    "Uniform": ((1.0, 4.0), "ss"),
    "Beta4": ((2.0, 3.0, 1.0, 5.0), "iiss"),
}


def _sample(name: str, censored: bool) -> tuple[Any, Any]:
    params, _ = FAMILIES[name]
    np.random.seed(3)
    x = np.asarray(getattr(surv, name).random(100, *params), dtype=float)
    c = np.zeros_like(x, dtype=int)
    if censored:
        cap = float(np.quantile(x, 0.85))
        c = (x > cap).astype(int)
        x = np.where(c == 1, cap, x)
    return x, c


def _cases() -> list[Any]:
    cases = []
    for name in FAMILIES:
        dist = getattr(surv, name)
        for how in ("MLE", "MPS", "MSE", "MOM", "MPP"):
            if how == "MPP" and not dist.supports_mpp:
                continue
            for censored in (False, True):
                if how == "MOM" and censored:
                    continue  # MOM takes complete data only
                label = f"{name}-{how}-{'cens' if censored else 'cplt'}"
                cases.append(pytest.param(name, how, censored, id=label))
    return cases


def _expected(params: Any, kinds: str, k: float) -> Any:
    out = {"s": lambda v: v * k, "r": lambda v: v / k}
    out.update({"l": lambda v: v + np.log(k), "i": lambda v: v})
    return np.array([out[t](v) for v, t in zip(params, kinds, strict=True)])


@pytest.mark.parametrize("name, how, censored", _cases())
def test_fit_is_equivariant_under_a_change_of_units(
    name: str, how: str, censored: bool
) -> None:
    dist = getattr(surv, name)
    _, kinds = FAMILIES[name]
    x, c = _sample(name, censored)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = dist.fit(x, c, how=how).params
        for k in SCALES:
            got = dist.fit(x * k, c, how=how).params
            want = _expected(ref, kinds, k)
            for g, w, t in zip(got, want, kinds, strict=True):
                # A log-location is already in relative units
                err = abs(g - w) if t == "l" else abs(g - w) / abs(w)
                assert err < TOL, (k, got, want)


@pytest.mark.parametrize("how", ["MLE", "MPS", "MSE", "MOM"])
@pytest.mark.parametrize("name", ["Normal", "Logistic", "Gumbel", "GumbelLEV"])
def test_a_location_near_zero_is_equivariant_too(name: str, how: str) -> None:
    # A location starting near 0 cannot take its search scale from its
    # own magnitude; ``search_floor`` supplies the data's spread instead.
    # At k = 1e-4 the old fixed floor of 1 left a GumbelLEV MSE fit 0.8%
    # (of its scale) away from the rescaled answer.
    dist = getattr(surv, name)
    np.random.seed(3)
    x = np.asarray(dist.random(100, 0.0, 2.0), dtype=float)
    x = x - np.median(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = dist.fit(x, how=how).params
        for k in SCALES:
            got = dist.fit(x * k, how=how).params / k
            # Relative to the scale: the location itself is near zero
            assert np.max(np.abs(got - ref)) / ref[1] < TOL, (k, got, ref)


def test_search_floor_follows_the_transformed_space() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 1e-3
    spread = float(np.std(x))
    # Unbounded endpoints get the data's spread; bounded shapes keep 1
    model = surv.Beta4.fit(x + 1e-2, how="MPS")
    np.testing.assert_allclose(search_floor(model), [1, 1, spread, spread])
    # ... capped at the old floor of 1 for data larger than that
    model = surv.Normal.fit(x * 1e4, how="MPS")
    np.testing.assert_allclose(search_floor(model), [1, 1])
    # A fixed parameter is not searched, so has no entry
    model = surv.Normal.fit(x, how="MPS", fixed={"sigma": 1e-3})
    np.testing.assert_allclose(search_floor(model), [spread])


def test_one_parameter_mom_mismatch_is_scale_free() -> None:
    # With one moment matched the mismatch was left in the data's squared
    # units, so at 1e-3 the start already met the tolerance: the fit
    # returned its starting point, 1.2% off.
    x, _ = _sample("Rayleigh", False)
    ref = surv.Rayleigh.fit(x, how="MOM").params
    for k in SCALES:
        got = surv.Rayleigh.fit(x * k, how="MOM").params
        np.testing.assert_allclose(got, ref * k, rtol=1e-6)
    # And the same for one free parameter of two
    x, _ = _sample("Weibull", False)
    ref = surv.Weibull.fit(x, how="MOM", fixed={"beta": 2.0}).params
    got = surv.Weibull.fit(x * 1e-3, how="MOM", fixed={"beta": 2.0}).params
    np.testing.assert_allclose(got, ref * [1e-3, 1], rtol=1e-6)


def test_uniform_initial_guess_scales_with_the_data() -> None:
    x = np.array([1.0, 2.0, 2.5, 4.0])
    data = surv.utils.surpyval_data.SurpyvalData(x=x)
    small = surv.utils.surpyval_data.SurpyvalData(x=x * 1e-3)
    init = surv.Uniform._parameter_initialiser(data)
    np.testing.assert_allclose(
        surv.Uniform._parameter_initialiser(small), init * 1e-3
    )
    # Strictly outside the data, so the likelihood is positive there
    assert init[0] < x.min() and init[1] > x.max()


@pytest.mark.parametrize("name", ["LogNormal", "ExpoWeibull"])
def test_mps_gradient_is_finite_at_the_support_edge(name: str) -> None:
    # With no truncation the left bound reaches MPS as the support's edge,
    # 0, where these CDFs have a singular derivative. The gradient was nan,
    # BFGS gave up and the fit ended on Nelder-Mead -- 1% short of the
    # optimum for a censored ExpoWeibull.
    x, c = _sample(name, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = getattr(surv, name).fit(x, c, how="MPS")
    assert model.optimizer == "BFGS"


# --------------------------------------------------------------------------
# Offset fits
#
# The offset gamma is a location: fitting ``k * x`` must put it at ``k``
# times the offset fitted to ``x``, with the other parameters mapping as
# above. Every offset search used to measure the offset's distance below
# the first observation in absolute units -- it started one unit below
# the data, was searched as ``log(min(x) - gamma)`` or linearly depending
# on whether that distance was below 1, and the probability-plot fit
# started its one-dimensional search at ``exp(0) = 1`` below the data --
# so at 1e-3 LogNormal, Gamma, Rayleigh and ExpoWeibull fits never moved
# from their start, and at 1e5 Weibull fits of every method, and all
# probability-plot fits, were wildly off.
# --------------------------------------------------------------------------

OFFSET_FAMILIES: dict[str, tuple[tuple[float, ...], str]] = {
    "Weibull": ((10.0, 2.0), "si"),
    "Exponential": ((0.1,), "r"),
    "Gamma": ((3.0, 0.5), "ir"),
    "LogNormal": ((1.5, 0.4), "li"),
    "LogLogistic": ((10.0, 4.0), "si"),
    "Rayleigh": ((3.0,), "s"),
    # A sample on which the offset fit is well posed: on some samples the
    # MPS and MSE optima run off along mu -> inf, where every scale finds a
    # different point on a ridge that is flat to the last digit.
    "ExpoWeibull": ((10.0, 3.0, 0.8), "sii"),
}
SHIFT = 10.0


def _offset_sample(name: str, censored: bool) -> tuple[Any, Any]:
    params, _ = OFFSET_FAMILIES[name]
    np.random.seed(5)
    x = SHIFT + np.asarray(getattr(surv, name).random(60, *params), float)
    c = np.zeros_like(x, dtype=int)
    if censored:
        cap = float(np.quantile(x, 0.85))
        c = (x > cap).astype(int)
        x = np.where(c == 1, cap, x)
    return x, c


def _offset_cases() -> list[Any]:
    cases = []
    for name in OFFSET_FAMILIES:
        dist = getattr(surv, name)
        for how in ("MLE", "MPS", "MSE", "MOM", "MPP"):
            if how == "MPP" and not dist.supports_mpp:
                continue
            for censored in (False, True):
                if how == "MOM" and censored:
                    continue  # MOM takes complete data only
                if name == "ExpoWeibull" and censored and how != "MLE":
                    # Ill posed on this sample: the MPS and MSE optima lie
                    # at mu -> inf, along a ridge flat to the last digit
                    continue
                label = f"{name}-{how}-{'cens' if censored else 'cplt'}"
                cases.append(pytest.param(name, how, censored, id=label))
    return cases


def _offset_scales(name: str, how: str) -> tuple[float, ...]:
    # The ExpoWeibull's offset moments are integrated numerically, at
    # seconds a fit: the two extremes only
    if name == "ExpoWeibull" and how == "MOM":
        return (1e-4, 1e5)
    return SCALES


@pytest.mark.parametrize("name, how, censored", _offset_cases())
def test_offset_fit_is_equivariant_under_a_change_of_units(
    name: str, how: str, censored: bool
) -> None:
    dist = getattr(surv, name)
    _, kinds = OFFSET_FAMILIES[name]
    x, c = _offset_sample(name, censored)
    spread = float(np.ptp(x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ref = dist.fit(x, c, how=how, offset=True)
        except ValueError:
            # A fit that cannot be made (a LogLogistic whose seeded shape
            # has no third moment, for MOM) cannot be made in any units
            for k in _offset_scales(name, how):
                with pytest.raises(ValueError):
                    dist.fit(x * k, c, how=how, offset=True)
            return
        for k in _offset_scales(name, how):
            got = dist.fit(x * k, c, how=how, offset=True)
            # The offset relative to the data's spread: it can be near 0
            assert abs(got.gamma / k - ref.gamma) / spread < TOL, (
                k,
                got.gamma / k,
                ref.gamma,
            )
            want = _expected(ref.params, kinds, k)
            for g, w, t in zip(got.params, want, kinds, strict=True):
                err = abs(g - w) if t == "l" else abs(g - w) / abs(w)
                assert err < TOL, (k, got.params, want)


def test_offset_start_is_a_step_on_the_data_scale() -> None:
    # The start was min(x) - 1 whatever the units; it is now the mean
    # spacing below the smallest value, so it moves with the data.
    x = np.array([5.0, 6.0, 8.0, 11.0])
    for k in (1e-4, 1.0, 1e5):
        init = surv.Weibull._initial_guess(
            surv.utils.surpyval_data.SurpyvalData(x=x * k),
            True,
            False,
            False,
            "Nelson-Aalen",
        )
        assert init[0] == pytest.approx((5.0 - 2.0) * k, rel=1e-12)


@pytest.mark.parametrize("name", ["LogNormal", "Gamma", "Rayleigh"])
def test_offset_seed_is_taken_against_the_installed_offset(name: str) -> None:
    # The initialisers seeded the other parameters from x - (min(x) - 1)
    # while the fitter installed a different starting offset; in data in
    # thousandths that was a thousand spreads away, and the seed with it.
    x, _ = _offset_sample(name, False)
    data = surv.utils.surpyval_data.SurpyvalData(x=x)
    small = surv.utils.surpyval_data.SurpyvalData(x=x * 1e-3)
    dist = getattr(surv, name)
    _, kinds = OFFSET_FAMILIES[name]
    init = dist._parameter_initialiser(data, offset=True)
    got = dist._parameter_initialiser(small, offset=True)
    assert got[0] == pytest.approx(init[0] * 1e-3, rel=1e-9)
    want = _expected(init[1:], kinds, 1e-3)
    np.testing.assert_allclose(got[1:], want, rtol=1e-6, atol=1e-9)


def test_lognormal_offset_mom_matches_three_moments_exactly() -> None:
    # The optimiser found the moment-matching offset LogNormal only from
    # some starting points: the mismatch has a spurious minimum with the
    # offset at the first observation, and three of five units ended
    # there (mismatch 0.19). It is now solved in closed form.
    x, _ = _offset_sample("LogNormal", False)
    model = surv.LogNormal.fit(x, how="MOM", offset=True)
    assert model.optimizer == "closed-form"
    centred = x - x.mean()
    assert model.mean() == pytest.approx(x.mean(), rel=1e-10)
    assert model.var() == pytest.approx(np.mean(centred**2), rel=1e-10)
    w = np.exp(model.params[1] ** 2)
    skew = np.mean(centred**3) / np.mean(centred**2) ** 1.5
    assert (w + 2) * np.sqrt(w - 1) == pytest.approx(skew, rel=1e-10)
