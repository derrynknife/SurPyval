"""
``MixtureModel.Hf`` and ``hf`` from the components' own cumulative hazards
and hazard rates: ``H = -log sum_i w_i exp(-H_i)`` (through ``-log1p`` of
the failed share while that is under a half) and ``h = sum_i pi_i h_i``
with ``pi_i = w_i S_i / S``. One call of each component's function, and
precise far into the upper tail.

The tail references are from ``scripts/reference/conditional_gaps_mpmath.py``
(mpmath at 50 digits), typed in here.
"""

from typing import Any

import numpy as np
import pytest

import surpyval as surv

# (mixture, x, Hf, hf) from scripts/reference/conditional_gaps_mpmath.py.
MIXTURE_TAIL = [
    ("weibull", 1.0, 0.00931662138077915, 0.007347495106030488),
    ("weibull", 50.0, 0.15626004120659737, 0.001703193670936798),
    ("weibull", 1000.0, 0.37604301793444667, 0.00030955887611774956),
    ("weibull", 5000.0, 24.992955912946023, 0.01729346571228401),
    ("weibull", 20000.0, 105.69078251218319, 0.004172179526042532),
    ("weibull", 100000.0, 379.3751535721291, 0.0030239108736880736),
    ("weibull", 220000.0, 711.6459465506352, 0.002582762371598238),
    ("lognormal", 0.5, 0.0007679583588670434, 0.005570285853149266),
    ("lognormal", 3.0, 0.3081492138920722, 0.16777849805796743),
    ("lognormal", 50.0, 2.008625723119809, 0.022274840318084826),
    ("lognormal", 10000.0, 16.499802739478508, 0.00044635152638684513),
    ("lognormal", 1e8, 86.55765843469185, 1.0772891440503833e-07),
    ("lognormal", 1e12, 215.10908709264788, 1.7145284504769083e-11),
    ("lognormal", 1e16, 402.42237542921157, 2.3530421268848937e-15),
    ("lognormal", 1e20, 648.5693118140808, 2.992020711247061e-19),
]


def _mixture(dist: Any, params: list, w: list) -> surv.MixtureModel:
    model = surv.MixtureModel(dist, m=len(w))
    model.params = np.array(params, dtype=float)
    model.w = np.array(w, dtype=float)
    return model


MIXTURES = {
    "weibull": _mixture(surv.Weibull, [[60, 0.8], [2000, 3.5]], [0.25, 0.75]),
    "lognormal": _mixture(surv.LogNormal, [[1, 0.5], [3, 1.2]], [0.4, 0.6]),
    "normal": _mixture(surv.Normal, [[0, 1], [5, 2]], [0.3, 0.7]),
    "gumbel": _mixture(surv.Gumbel, [[0, 1], [5, 2]], [0.3, 0.7]),
    "gamma": _mixture(surv.Gamma, [[2, 1], [5, 0.3]], [0.5, 0.5]),
    "exponential": _mixture(surv.Exponential, [[1], [0.01]], [0.9, 0.1]),
    "three": _mixture(
        surv.Weibull, [[5, 0.7], [50, 2], [500, 6]], [0.2, 0.5, 0.3]
    ),
}


def _from_survival(model: surv.MixtureModel, x: np.ndarray) -> tuple:
    """The formulas the mixture used before: ``-log sf`` (``-log1p(-ff)``
    while ``ff < 1/2``) and ``df / sf``, from the summed functions."""
    with np.errstate(all="ignore"):
        F = sum(
            w * model.dist.ff(x, *p) for w, p in zip(model.w, model.params)
        )
        S = sum(
            w * model.dist.sf(x, *p) for w, p in zip(model.w, model.params)
        )
        f = sum(
            w * model.dist.df(x, *p) for w, p in zip(model.w, model.params)
        )
        return np.where(F < 0.5, -np.log1p(-F), -np.log(S)), f / S, S


@pytest.mark.parametrize("name,x,H,h", MIXTURE_TAIL)
def test_mixture_hazards_match_mpmath_into_the_far_tail(name, x, H, h):
    # To 1e-12 relative, out to H ~ 700 (a survival of 1e-309, where the
    # summed survival underflows). The hazard rate goes through
    # exp(H - H_i), so an absolute error of eps * H in H is a relative
    # error of about that in h: 1.5e-13 at H = 700.
    model = MIXTURES[name]
    assert model.Hf(x) == pytest.approx(H, rel=1e-12)
    assert model.hf(x) == pytest.approx(h, rel=1e-12)


@pytest.mark.parametrize("name", sorted(MIXTURES))
def test_mixture_hazards_match_the_survival_formulas_in_the_body(name):
    model = MIXTURES[name]
    x = np.concatenate(
        [np.geomspace(1e-4, 1e5, 300), -np.geomspace(1e-3, 1e2, 30)]
    )
    H_old, h_old, S = _from_survival(model, x)
    # Where the old formulas are themselves accurate (the summed survival
    # well clear of underflow), to 1e-12 relative.
    body = S > 1e-100
    np.testing.assert_allclose(model.Hf(x)[body], H_old[body], rtol=1e-12)
    np.testing.assert_allclose(
        model.hf(x)[body], h_old[body], rtol=1e-12, atol=1e-300
    )


@pytest.mark.parametrize("name", sorted(MIXTURES))
def test_mixture_hf_is_the_derivative_of_Hf(name):
    model = MIXTURES[name]
    x = np.geomspace(0.05, 200.0, 40)
    H = model.Hf(x)
    x = x[(H > 1e-6) & (H < 500)]
    step = 1e-6 * x
    slope = (model.Hf(x + step) - model.Hf(x - step)) / (2 * step)
    # Central differences: an error of about step^2 * H''' / 6 plus the
    # rounding eps * H / step, well under 1e-5 relative here.
    np.testing.assert_allclose(model.hf(x), slope, rtol=1e-5)


def test_mixture_hazards_at_the_edges():
    model = MIXTURES["weibull"]
    x = np.array([-1.0, 0.0, np.nan, np.inf])
    H = model.Hf(x)
    assert H[0] == 0 and H[1] == 0 and np.isnan(H[2]) and H[3] == np.inf
    h = model.hf(x)
    assert h[0] == 0 and np.isnan(h[2])
    # Shapes are kept as the other methods keep them.
    assert np.ndim(model.Hf(5.0)) == 0
    assert model.hf([[1.0, 2.0]]).shape == (1, 2)


class _Counting:
    """A distribution that counts the calls of its functions."""

    def __init__(self, dist: Any, drop: tuple = ()) -> None:
        self._dist = dist
        self._drop = drop
        self.calls: dict = {}

    def __getattr__(self, name: str) -> Any:
        if name in self._drop:
            raise AttributeError(name)
        value = getattr(self._dist, name)
        if not callable(value):
            return value

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls[name] = self.calls.get(name, 0) + 1
            return value(*args, **kwargs)

        return counted


def test_mixture_hazards_call_each_component_once():
    model = _mixture(surv.Weibull, [[60, 0.8], [2000, 3.5]], [0.25, 0.75])
    counting = _Counting(surv.Weibull)
    model.dist = counting
    x = np.array([10.0, 5000.0, 2e5])
    model.Hf(x)
    assert counting.calls == {"Hf": 2}
    counting.calls.clear()
    model.hf(x)
    assert counting.calls == {"Hf": 2, "hf": 2}


def test_mixture_without_component_hazards_uses_the_survival_formulas():
    reference = MIXTURES["weibull"]
    model = _mixture(surv.Weibull, [[60, 0.8], [2000, 3.5]], [0.25, 0.75])
    model.dist = _Counting(surv.Weibull, drop=("hf",))
    x = np.array([10.0, 500.0, 5000.0])
    np.testing.assert_allclose(model.hf(x), reference.hf(x), rtol=1e-12)
    assert "hf" not in model.dist.calls and model.dist.calls.get("df")
