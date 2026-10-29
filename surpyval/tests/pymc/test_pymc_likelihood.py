"""The pytensor likelihood of ``surpyval.pymc`` against SurPyval's own.

At fixed parameters, ``log_likelihood`` must equal the distribution's
``_log_likelihood`` (what ``fit`` maximises) for every supported family,
every row type and every truncation pattern, to 1e-10. Its gradient must
agree with a central difference of SurPyval's. Only pytensor is needed;
the module is skipped where it is not installed (PyMC is not a test
dependency of SurPyval).
"""

import numpy as np
import pytest

import surpyval as sp

pt = pytest.importorskip("pytensor.tensor")
pytensor = pytest.importorskip("pytensor")

from surpyval.pymc import SUPPORTED, log_likelihood  # noqa: E402

INF = np.inf

# Parameters at which each family puts its mass over the times below
# (between 1 and 30), and where some windows lie in its upper tail.
PARAMS = {
    "Weibull": [10.0, 1.7],
    "Exponential": [0.09],
    "LogNormal": [2.2, 0.6],
    "Gamma": [2.5, 0.25],
    "LogLogistic": [9.0, 2.5],
    "Normal": [11.0, 6.0],
    "Gumbel": [14.0, 5.0],
    "Logistic": [11.0, 3.5],
    "Rayleigh": [9.0],
}

# Each row type alone, then mixed with counts. Interval [22, 27] sits in
# every family's upper tail (F(22) > 1/2), where the window is taken
# from the survival function.
ROWS = {
    "observed": dict(x=[2.0, 5.5, 9.0, 13.0, 21.0]),
    "right": dict(x=[3.0, 8.0, 15.0], c=[1, 1, 1]),
    "left": dict(x=[4.0, 7.5, 12.0], c=[-1, -1, -1]),
    "interval": dict(x=[[2.0, 6.0], [8.0, 9.0], [22.0, 27.0]], c=[2, 2, 2]),
    "mixed": dict(
        x=[2.0, 5.0, [6.0, 10.0], 9.0, 14.0, [22.0, 27.0], 18.0, 4.5],
        c=[0, 1, 2, -1, 0, 2, 1, -1],
        n=[1, 3, 2, 1, 4, 1, 2, 5],
    ),
}

# Truncation patterns, as (tl, tr) for every row.
TRUNCATION = {
    "none": (None, None),
    "left": (1.0, None),
    "right": (None, 29.0),
    "both": (1.0, 29.0),
    "left_at_support_edge": (0.0, 29.0),
}


def _arrays(rows, truncation):
    arrays = {k: list(v) for k, v in ROWS[rows].items()}
    tl, tr = TRUNCATION[truncation]
    k = len(arrays["x"])
    if tl is not None:
        arrays["tl"] = [tl] * k
    if tr is not None:
        arrays["tr"] = [tr] * k
    return arrays


def _surpyval_ll(dist, arrays, params):
    data = sp.SurpyvalData(**arrays)
    return float(dist._log_likelihood(data, *params, 0.0, 0.0, 1.0))


def _dist(name):
    return getattr(sp, name)


def _kw(dist, params):
    return dict(zip(dist.parameter_names, params))


def test_every_supported_family_is_tested():
    assert set(PARAMS) == set(SUPPORTED)


@pytest.mark.parametrize("truncation", list(TRUNCATION))
@pytest.mark.parametrize("rows", list(ROWS))
@pytest.mark.parametrize("name", list(PARAMS))
def test_log_likelihood_equals_surpyval(name, rows, truncation):
    dist = _dist(name)
    if truncation == "left_at_support_edge" and dist.support[0] != 0:
        pytest.skip("only a positive family has its support edge at 0")
    params = PARAMS[name]
    arrays = _arrays(rows, truncation)
    expected = _surpyval_ll(dist, arrays, params)
    got = float(log_likelihood(dist, **arrays, **_kw(dist, params)).eval())
    assert np.isfinite(expected)
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("name", list(PARAMS))
def test_a_surpyval_data_gives_the_same_term(name):
    dist = _dist(name)
    params = PARAMS[name]
    arrays = _arrays("mixed", "both")
    data = sp.SurpyvalData(**arrays)
    got = float(log_likelihood(dist, data, **_kw(dist, params)).eval())
    assert got == pytest.approx(
        _surpyval_ll(dist, arrays, params), rel=1e-10, abs=1e-10
    )


@pytest.mark.parametrize("name", list(PARAMS))
def test_per_row_parameters_give_each_row_its_own(name):
    # A parameter given per row equals the sum of each row's likelihood
    # at its own value: what a covariate model relies on.
    dist = _dist(name)
    base = PARAMS[name]
    arrays = _arrays("mixed", "both")
    k = len(arrays["x"])
    scale = np.linspace(0.85, 1.15, k)
    first = base[0] * scale
    rest = {p: v for p, v in zip(dist.parameter_names[1:], base[1:])}
    got = float(
        log_likelihood(
            dist, **arrays, **{dist.parameter_names[0]: first}, **rest
        ).eval()
    )
    expected = 0.0
    for i in range(k):
        row = {key: [value[i]] for key, value in arrays.items()}
        expected += _surpyval_ll(dist, row, [first[i], *base[1:]])
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("name", list(PARAMS))
def test_gradient_matches_a_central_difference(name):
    dist = _dist(name)
    params = np.array(PARAMS[name])
    arrays = _arrays("mixed", "both")
    theta = [pt.dscalar(p) for p in dist.parameter_names]
    ll = log_likelihood(
        dist, **arrays, **dict(zip(dist.parameter_names, theta))
    )
    grad = pytensor.function(theta, pt.grad(ll, theta))(*params)
    for i in range(len(params)):
        h = 1e-6 * abs(params[i])
        up, down = params.copy(), params.copy()
        up[i] += h
        down[i] -= h
        fd = (
            _surpyval_ll(dist, arrays, up) - _surpyval_ll(dist, arrays, down)
        ) / (2 * h)
        assert np.isfinite(grad[i])
        assert grad[i] == pytest.approx(fd, rel=1e-5, abs=1e-6)


def test_far_upper_tail_window_stays_finite():
    # A left truncation far in the upper tail, where F(tl) rounds to 1
    # and 1 - F would be 0: taken from log S, as SurPyval does.
    arrays = dict(x=[60.0, 70.0], c=[0, 1], tl=[50.0, 50.0], tr=[90.0, 90.0])
    expected = _surpyval_ll(sp.Weibull, arrays, [5.0, 2.0])
    got = float(
        log_likelihood(sp.Weibull, **arrays, alpha=5.0, beta=2.0).eval()
    )
    assert np.isfinite(got)
    assert got == pytest.approx(expected, rel=1e-10)


def test_xl_xr_and_t_forms():
    xl, xr = [1.0, 4.0, 6.0], [3.0, 4.0, 9.0]
    t = [[0.5, 20.0]] * 3
    expected = _surpyval_ll(sp.Weibull, dict(xl=xl, xr=xr, t=t), [6.0, 1.5])
    got = log_likelihood(sp.Weibull, xl=xl, xr=xr, t=t, alpha=6.0, beta=1.5)
    assert float(got.eval()) == pytest.approx(expected, rel=1e-10)


def test_an_array_as_data_is_x():
    a = log_likelihood(sp.Exponential, [1.0, 2.0, 5.0], failure_rate=0.3)
    b = log_likelihood(sp.Exponential, x=[1.0, 2.0, 5.0], failure_rate=0.3)
    assert float(a.eval()) == float(b.eval())


# Refusals: they read as fit's.


@pytest.mark.parametrize(
    "arrays",
    [
        dict(x=[1.0, 2.0], c=[0, 3]),
        dict(x=[1.0, np.nan]),
        dict(x=[1.0, 2.0], n=[1, 0]),
        dict(x=[1.0, 5.0], tl=[2.0, 2.0]),
    ],
)
def test_invalid_data_is_refused_as_fit_refuses_it(arrays):
    with pytest.raises(ValueError) as fit_err:
        sp.Weibull.fit(**arrays)
    with pytest.raises(ValueError) as ours:
        log_likelihood(sp.Weibull, **arrays, alpha=1.0, beta=1.0)
    assert str(ours.value) == str(fit_err.value)


def test_data_outside_the_support_is_refused_as_fit_refuses_it():
    with pytest.raises(ValueError) as fit_err:
        sp.Weibull.fit([-1.0, 2.0, 3.0])
    with pytest.raises(ValueError) as ours:
        log_likelihood(sp.Weibull, x=[-1.0, 2.0, 3.0], alpha=1.0, beta=1.0)
    assert str(ours.value) == str(fit_err.value)


def test_an_unsupported_family_is_refused_with_the_supported_list():
    with pytest.raises(ValueError, match="supports Weibull, Exponential"):
        log_likelihood(sp.Beta, x=[0.2, 0.5], alpha=1.0, beta=1.0)


def test_a_fitted_model_is_refused():
    model = sp.Weibull.fit([1.0, 2.0, 4.0, 7.0])
    with pytest.raises(TypeError, match="not a fitted model"):
        log_likelihood(model, x=[1.0, 2.0], alpha=1.0, beta=1.0)


def test_missing_and_unknown_parameters_are_refused():
    with pytest.raises(TypeError, match="missing beta"):
        log_likelihood(sp.Weibull, x=[1.0, 2.0], alpha=1.0)
    with pytest.raises(TypeError, match="no parameter 'shape'"):
        log_likelihood(sp.Weibull, x=[1.0, 2.0], alpha=1.0, beta=1, shape=2)


def test_a_per_row_parameter_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="2 values and the data 3 rows"):
        log_likelihood(
            sp.Weibull, x=[1.0, 2.0, 3.0], alpha=[1.0, 2.0], beta=1.0
        )
    # A length only known when the graph runs is checked then.
    v = pt.dvector("alpha")
    ll = log_likelihood(sp.Weibull, x=[1.0, 2.0, 3.0], alpha=v, beta=1.0)
    with pytest.raises(AssertionError, match="one value per data row"):
        ll.eval({v: np.array([1.0, 2.0])})


def test_data_given_twice_is_refused():
    data = sp.SurpyvalData(x=[1.0, 2.0])
    with pytest.raises(TypeError, match="leave out 'c'"):
        log_likelihood(sp.Weibull, data, c=[0, 1], alpha=1.0, beta=1.0)
