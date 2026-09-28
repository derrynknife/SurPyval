"""``NonParametric.set_bounds``: an explicit support for the estimate.

Without it (``support is None``, what a fit gives) nothing changes. With
it, every function and every ``interp`` is at its start value in
``[lower, x[0])``, carries its value at ``x[-1]`` to ``upper``, and is NaN
outside ``[lower, upper]`` (principle 11); the confidence bounds follow.
"""

import copy
import json
import warnings

import numpy as np
import pytest

import surpyval
from surpyval import FlemingHarrington, KaplanMeier, NelsonAalen, Turnbull

ESTIMATORS = {
    "KaplanMeier": KaplanMeier,
    "NelsonAalen": NelsonAalen,
    "FlemingHarrington": FlemingHarrington,
    "Turnbull": Turnbull,
}
INTERPS = ("step", "linear", "cubic", "nearest", "previous", "slinear")
FUNCTIONS = ("sf", "ff", "Hf", "hf", "df")
START = {"sf": 1.0, "ff": 0.0, "Hf": 0.0, "hf": 0.0, "df": 0.0}

# Right censored last, so no estimator reaches 0 (a Kaplan-Meier at 0 has
# an infinite hazard jump, and df there is NaN with a warning, #408).
X = np.array([2.0, 3, 3, 5, 6, 8, 9, 11, 12, 14])
C = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1])


def _fit(name, x=X, c=C):
    return ESTIMATORS[name].fit(x, c=c)


def _bounded(model, lower, upper):
    return copy.deepcopy(model).set_bounds(lower, upper)


def _is_clean_zero(v):
    return np.all(v == 0) and not np.any(np.signbit(v))


@pytest.fixture(params=list(ESTIMATORS))
def model(request):
    return _fit(request.param)


def test_a_fit_has_no_support(model):
    assert model.support is None
    assert KaplanMeier.from_xrd([1, 2], [2, 1], [1, 1]).support is None
    ecdf = surpyval.NonParametric.fit_from_ecdf([1, 2], [0.5, 0.2])
    assert ecdf.support is None


def test_set_bounds_returns_the_model_and_stores_floats(model):
    out = model.set_bounds(0, 20)
    assert out is model
    assert model.support == (0.0, 20.0)
    assert all(type(v) is float for v in model.support)
    # Setting again replaces the bounds.
    assert model.set_bounds(-np.inf, np.inf).support == (-np.inf, np.inf)


@pytest.mark.parametrize("interp", INTERPS)
@pytest.mark.parametrize("fname", FUNCTIONS)
def test_every_region(model, fname, interp):
    first, last = model.x[0], model.x[-1]
    lower, upper = first - 2.0, last + 5.0
    bounded = _bounded(model, lower, upper)

    def f(m, q):
        return np.asarray(getattr(m, fname)(q, interp=interp), float)

    # Inside the data: exactly the unbounded values.
    inside = np.unique(np.concatenate([model.x, model.x[:-1] + 0.5]))
    np.testing.assert_array_equal(f(bounded, inside), f(model, inside))

    # Below lower and above upper, and a missing value: NaN.
    for q in ([lower - 1.0], [upper + 1.0], [np.nan], [-np.inf], [np.inf]):
        assert np.isnan(f(bounded, q)).all(), (q, f(bounded, q))

    # [lower, first): the start value, a clean 0.0 where it is 0.
    below = np.array([lower, lower + 1.0, first - 1e-9])
    v = f(bounded, below)
    np.testing.assert_array_equal(v, START[fname])
    if START[fname] == 0.0:
        assert _is_clean_zero(v)

    # (last, upper]: the value at the last point carried (hf and df: the
    # step containing it, i.e. the value the query's last point has).
    query = np.concatenate([inside, [last + 1e-9, last + 2.0, upper]])
    v = f(bounded, query)
    at_last = v[inside.size - 1]
    np.testing.assert_array_equal(v[inside.size :], at_last)
    if fname in ("sf", "ff", "Hf"):
        np.testing.assert_array_equal(at_last, f(model, [last])[0])
        # Also as scalars.
        assert f(bounded, upper)[0] == f(model, last)[0]


@pytest.mark.parametrize("interp", INTERPS)
def test_infinite_bounds_take_infinite_queries(model, interp):
    bounded = _bounded(model, -np.inf, np.inf)
    q = np.array([-np.inf, np.inf])
    np.testing.assert_array_equal(
        bounded.sf(q, interp=interp), [1.0, model.R[-1]]
    )
    assert bounded.sf(np.inf, interp=interp)[0] == model.R[-1]
    assert bounded.sf(-np.inf, interp=interp)[0] == 1.0
    assert _is_clean_zero(bounded.Hf([-np.inf], interp=interp))
    assert bounded.Hf([np.inf], interp=interp)[0] == -np.log(model.R[-1])
    # Without bounds a step sf is 0 at inf, and the interpolated ones NaN:
    # unchanged.
    if interp == "step":
        assert model.sf(np.inf)[0] == 0.0
    else:
        assert np.isnan(model.sf(np.inf, interp=interp)[0])


def test_a_query_mixing_every_region_keeps_its_order(model):
    first, last = model.x[0], model.x[-1]
    bounded = _bounded(model, first - 1, last + 1)
    q = np.array([last + 0.5, np.nan, first - 0.5, 100.0, (first + last) / 2])
    v = bounded.sf(q, interp="linear")
    assert v[0] == model.R[-1]
    assert np.isnan(v[1]) and np.isnan(v[3])
    assert v[2] == 1.0
    assert v[4] == model.sf((first + last) / 2, interp="linear")[0]
    # A 2-D query keeps its shape.
    assert bounded.sf(q.reshape(5, 1)).shape == (5, 1)
    # An empty query gives an empty result.
    assert bounded.sf(np.array([])).shape == (0,)


def test_the_interpolated_forms_are_nan_outside_the_data_without_bounds():
    # What set_bounds changes (the step forms already hold their values).
    model = _fit("KaplanMeier")
    q = [X[0] - 1, X[-1] + 1]
    for interp in INTERPS[1:]:
        assert np.isnan(model.sf(q, interp=interp)).all()
        bounded = _bounded(model, 0, 20)
        np.testing.assert_array_equal(
            bounded.sf(q, interp=interp), [1.0, model.R[-1]]
        )


def test_hf_before_the_first_value_is_zero_and_outside_points_are_ignored():
    model = _fit("NelsonAalen")
    bounded = _bounded(model, 0, 20)
    # A point outside the bounds takes no part in the differences.
    q = np.array([2.5, 4.0, 5.5, 7.0])
    with_outside = np.array([-5.0, 2.5, 4.0, 5.5, 7.0, 30.0])
    v = bounded.hf(with_outside)
    assert np.isnan(v[[0, -1]]).all()
    np.testing.assert_array_equal(v[1:-1], model.hf(q))
    # Before the first value the increment is 0, for arrays and scalars.
    assert _is_clean_zero(bounded.hf([0.5, 1.0]))
    assert _is_clean_zero(bounded.hf(1.0))
    assert _is_clean_zero(bounded.df(1.0))
    # Unbounded, the increment before the first failure is NaN.
    assert np.isnan(model.hf(1.0)).all()


@pytest.mark.parametrize("bound_type", ["exp", "normal"])
@pytest.mark.parametrize("interp", ["step", "linear", "cubic"])
@pytest.mark.parametrize("bound", ["two-sided", "upper", "lower"])
@pytest.mark.parametrize("on", ["sf", "ff", "Hf"])
def test_cb_regions(model, on, bound, interp, bound_type):
    first, last = model.x[0], model.x[-1]
    lower, upper = first - 2.0, last + 5.0
    bounded = _bounded(model, lower, upper)
    kw = {"on": on, "bound": bound, "interp": interp, "bound_type": bound_type}

    inside = np.unique(np.concatenate([model.x, model.x[:-1] + 0.5]))
    np.testing.assert_array_equal(
        bounded.cb(inside, **kw), model.cb(inside, **kw)
    )
    # Without bounds, NaN outside the data; with them, only outside them.
    assert np.isnan(model.cb([first - 1.0, last + 1.0], **kw)).all()
    assert np.isnan(bounded.cb([lower - 1.0, upper + 1.0, np.nan], **kw)).all()
    # The start value, with no width, before the first value.
    start = bounded.cb([lower, first - 1.0], **kw)
    np.testing.assert_array_equal(start, 1.0 if on == "sf" else 0.0)
    if on != "sf":
        assert _is_clean_zero(start)
    # The bounds at the last value, carried.
    after = bounded.cb([last + 1.0, upper], **kw)
    at_last = model.cb([last], **kw)
    np.testing.assert_array_equal(after, np.concatenate([at_last, at_last]))
    # They still contain the estimate.
    if bound == "two-sided" and bound_type == "exp":
        q = np.array([lower, first - 1.0, last + 1.0, upper])
        est = getattr(bounded, on)(q, interp=interp)
        cb = bounded.cb(q, **kw)
        assert np.all(cb[:, 0] <= est + 1e-12)
        assert np.all(est <= cb[:, 1] + 1e-12)


def test_R_cb_and_bootstrap_cb_follow_the_bounds():
    model = _fit("KaplanMeier")
    bounded = _bounded(model, 0, 20)
    q = np.array([-1.0, 1.0, 3.0, 9.5, 14.0, 17.0, 21.0, np.nan])
    r = bounded.R_cb(q)
    assert np.isnan(r[[0, -2, -1]]).all()
    np.testing.assert_array_equal(r[1], [1.0, 1.0])
    np.testing.assert_array_equal(r[2:5], model.R_cb(q[2:5]))
    np.testing.assert_array_equal(r[5], model.R_cb([14.0])[0])

    boot = bounded.bootstrap_cb(q, B=30, random_state=2)
    unbounded = model.bootstrap_cb(q[2:5], B=30, random_state=2)
    assert np.isnan(boot[[0, -2, -1]]).all()
    np.testing.assert_array_equal(boot[1], [1.0, 1.0])
    np.testing.assert_array_equal(boot[2:5], unbounded)
    np.testing.assert_array_equal(boot[5], boot[4])
    one_sided = bounded.bootstrap_cb(q, bound="lower", B=30, random_state=2)
    assert one_sided.shape == (q.size,)
    assert np.isnan(one_sided[[0, -2, -1]]).all()


def test_band_quantile_cb_and_smoothed_hf_are_unchanged():
    model = _fit("KaplanMeier")
    bounded = _bounded(model, 0, 20)
    q = np.array([-1.0, 1.0, 3.0, 9.5, 14.0, 17.0, 21.0])
    np.testing.assert_array_equal(bounded.band(q), model.band(q))
    np.testing.assert_array_equal(
        bounded.quantile_cb([0.2, 0.5]), model.quantile_cb([0.2, 0.5])
    )
    np.testing.assert_array_equal(
        bounded.smoothed_hf(q, bandwidth=3), model.smoothed_hf(q, bandwidth=3)
    )
    np.testing.assert_array_equal(bounded.qf([0.2, 0.5]), model.qf([0.2, 0.5]))
    assert bounded.mean() == model.mean()


def test_no_raw_numpy_warning(model):
    bounded = _bounded(model, -np.inf, 30)
    q = np.array([-np.inf, -3.0, 1.0, 2.0, 4.5, 14.0, 20.0, 30.0, 31.0])
    q = np.append(q, [np.inf, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for interp in INTERPS:
            for fname in FUNCTIONS:
                getattr(bounded, fname)(q, interp=interp)
                getattr(bounded, fname)(q[5], interp=interp)
            for on in ("sf", "ff", "Hf"):
                bounded.cb(q, on=on, interp=interp)
        bounded.bootstrap_cb(q, B=5, random_state=1)


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_negative_values(name):
    # The variable need not be time: every estimator fits negative values,
    # and the bounds may be negative.
    x = np.array([-7.5, -5.0, -3.0, -1.0, 0.0, 2.0])
    c = np.array([0, 1, 0, 0, 0, 1])
    model = _fit(name, x, c)
    bounded = _bounded(model, -10.0, 5.0)
    q = np.array([-11.0, -10.0, -8.0, -4.0, 2.0, 3.0, 5.0, 6.0])
    sf = bounded.sf(q, interp="linear")
    assert np.isnan(sf[[0, -1]]).all()
    np.testing.assert_array_equal(sf[1:3], 1.0)
    assert sf[3] == model.sf(-4.0, interp="linear")[0]
    np.testing.assert_array_equal(sf[4:7], model.R[-1])
    assert _is_clean_zero(bounded.ff([-10.0, -8.0]))
    with pytest.raises(ValueError, match="-7.5"):
        _bounded(model, -7.0, 5.0)


def test_set_lower_limit_is_the_first_value():
    model = KaplanMeier.fit(X, c=C, set_lower_limit=0)
    assert model.x[0] == 0
    bounded = _bounded(model, -5, 20)
    np.testing.assert_array_equal(bounded.sf([-5, -1, 0, 1]), 1.0)
    with pytest.raises(ValueError, match="at most 0"):
        _bounded(model, 1, 20)


@pytest.mark.parametrize(
    "lower, upper, match",
    [
        (3.0, 20.0, r"'lower' \(3.0\) is above the first value .*\(2.0\)"),
        (0.0, 13.0, r"'upper' \(13.0\) is below its last value \(14.0\)"),
        (20.0, 0.0, "'lower' must be below 'upper'"),
        (np.inf, np.inf, "'lower' must be below 'upper'"),
        (np.nan, 20.0, "'lower' must be a number, not NaN"),
        (0.0, np.nan, "'upper' must be a number, not NaN"),
        ("a", 20.0, "'lower' must be a number; got 'a'"),
        (0.0, None, "'upper' must be a number; got None"),
    ],
)
def test_invalid_bounds_are_refused(lower, upper, match):
    model = _fit("KaplanMeier")
    with pytest.raises(ValueError, match=match):
        model.set_bounds(lower, upper)
    # A refused call leaves the model as it was.
    assert model.support is None


def test_bounds_at_the_data_are_allowed():
    model = _fit("KaplanMeier")
    bounded = _bounded(model, X[0], X[-1])
    assert bounded.support == (2.0, 14.0)
    assert np.isnan(bounded.sf([1.9, 14.1])).all()
    np.testing.assert_array_equal(bounded.sf(X), model.sf(X))


def _all_values(model):
    q = np.array([-5.0, 0.0, 1.0, 2.0, 4.5, 14.0, 20.0, 31.0, np.inf])
    out = [getattr(model, f)(q, interp=i) for f in FUNCTIONS for i in INTERPS]
    out += [model.cb(q, on=on) for on in ("sf", "ff", "Hf")]
    return out


@pytest.mark.parametrize(
    "support, schema", [((0.0, 30.0), 2), ((-np.inf, np.inf), 2)]
)
def test_serialisation_round_trip(model, support, schema):
    bounded = _bounded(model, *support)
    d = bounded.to_dict()
    assert d["schema"] == schema
    assert "support" in d
    text = json.dumps(d, allow_nan=False)
    restored = surpyval.from_dict(json.loads(text))
    assert restored.support == support
    for a, b in zip(_all_values(bounded), _all_values(restored)):
        np.testing.assert_array_equal(a, b)
    # The class reader too.
    assert surpyval.NonParametric.from_dict(json.loads(text)).support == (
        support
    )


def test_without_bounds_the_dictionary_is_unchanged():
    # A Kaplan-Meier whose curve stays above 0 has no non-finite value, so
    # it is still readable by v0.20 (schema 1).
    model = _fit("KaplanMeier")
    d = model.to_dict()
    assert "support" not in d
    assert d["schema"] == 1
    assert surpyval.from_dict(d).support is None


def test_a_corrupt_support_is_refused():
    d = _bounded(_fit("KaplanMeier"), 0, 20).to_dict()
    with pytest.raises(ValueError, match=r"\[lower, upper\] pair"):
        surpyval.from_dict({**d, "support": [0.0]})
    with pytest.raises(ValueError, match="'lower' .* is above"):
        surpyval.from_dict({**d, "support": [5.0, 20.0]})


def test_cubic_sf_at_the_last_knot_is_not_below_zero():
    # PCHIP round-off at the last knot of a Kaplan-Meier that ends at 0
    # gave sf = -2.3e-17, and so Hf = NaN with a raw "invalid value in
    # log" warning (found by the set_bounds conformance check, which
    # queries the last time).
    x = [2.411, 3.84, 4.956, 5.953, 6.903, 7.846, 8.816, 9.85, 10.997]
    x = np.array(x + [12.347, 14.096, 16.954])
    c = np.zeros(12, int)
    c[[4, 9]] = 1
    n = np.ones(12, int)
    n[[2, 7]] = [2, 3]
    model = KaplanMeier.fit(x, c=c, n=n)
    assert model.R[-1] == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sf = model.sf(model.x, interp="cubic")
        H = model.Hf(model.x[-1], interp="cubic")
    assert np.all((sf >= 0) & (sf <= 1))
    assert sf[-1] == 0.0 and not np.signbit(sf[-1])
    assert H[0] == np.inf
