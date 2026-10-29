"""The properties every fit with sampling weights ``w`` keeps.

Every public fitter whose ``fit`` takes ``w`` is registered here once, in
``WEIGHTED``, with the bounds it offers for weighted data (``offered``)
and those it refuses (``refused``), the every-bound rule's declaration
for weighted fits; ``test_every_fitter_taking_w_is_registered`` fails for
one that is not. Each is checked for:

1. ``w = 1`` everywhere gives the estimates of no ``w`` (the intervals are
   then the robust ones, not the classical ones);
2. multiplying every weight by a constant (0.1, 10) changes nothing:
   estimates and intervals (the property the classical variances lack);
3. the order of the rows does not matter;
4. counts still equal repetition: a row with ``n = 2`` and weight ``w``
   is two rows with ``n = 1`` and weight ``w``;
5. each offered bound is finite; each refused
   bound raises a ``WeightedRefusal`` naming what to use instead;
6. the information criteria refuse;
7. ``to_dict`` / ``from_dict`` (stamped schema 3) and ``pickle`` keep
   the robust intervals and the refusals.
"""

import inspect
import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import quietly, weighted_reference_data
from surpyval.utils.weights import WeightedRefusal

QUERY = np.array([1.5, 3.0, 6.0])


@dataclass
class WeightedCase:
    name: str
    #: fit(x, c, n, w, Z, tl) -> model
    fit: Callable[..., Any]
    #: name -> model -> value: the estimates compared between fits
    estimates: Callable[[Any], np.ndarray]
    offered: dict = field(default_factory=dict)
    refused: dict = field(default_factory=dict)
    covariates: bool = False
    #: The tolerance of the bounds and covariance between fits that reach
    #: the same maximum: a regression's covariance is a numerical Hessian
    #: where the exact one was not kept, good to about 1e-3.
    rtol: float = 1e-4


def _np(fitter):
    return WeightedCase(
        fitter,
        lambda x, c, n, w, Z, tl: getattr(sp, fitter).fit(x, c, n, w=w, tl=tl),
        lambda m: m.sf(QUERY),
        offered={
            "cb(sf)": lambda m: m.cb(QUERY, on="sf"),
            "cb(Hf)": lambda m: m.cb(QUERY, on="Hf"),
            "cb(normal)": lambda m: m.cb(QUERY, bound_type="normal"),
            "quantile_cb": lambda m: m.quantile_cb(0.3),
            "rmst": lambda m: np.array(
                [m.rmst(6.0)["lower"], m.rmst(6.0)["upper"]]
            ),
        },
        refused={
            "band": lambda m: m.band(),
            "bootstrap_cb": lambda m: m.bootstrap_cb(QUERY, random_state=1),
        },
    )


def _parametric(dist):
    return WeightedCase(
        dist,
        lambda x, c, n, w, Z, tl: getattr(sp, dist).fit(x, c, n, w=w, tl=tl),
        lambda m: np.asarray(m.params, dtype=float),
        offered={
            "cb(sf)": lambda m: m.cb(QUERY, on="sf"),
            "cb(Hf)": lambda m: m.cb(QUERY, on="Hf"),
            "param_cb": lambda m: m.param_cb(m.dist.parameter_names[0]),
            "quantile_cb": lambda m: m.quantile_cb(0.3),
        },
        refused={
            "cb(lr)": lambda m: m.cb(QUERY, method="lr"),
            "cb(bootstrap)": lambda m: m.cb(
                QUERY, method="bootstrap", random_state=1
            ),
            "param_cb(lr)": lambda m: m.param_cb(
                m.dist.parameter_names[0], method="lr"
            ),
            "plot(Blom)": lambda m: m.get_plot_data(heuristic="Blom"),
        },
    )


def _regression(fitter):
    return WeightedCase(
        fitter,
        lambda x, c, n, w, Z, tl: getattr(sp, fitter).fit(
            x, Z, c=c, n=n, w=w, tl=tl
        ),
        lambda m: np.asarray(m.params, dtype=float),
        offered={
            "cb(sf)": lambda m: m.cb(QUERY, np.zeros((3, 2)), on="sf"),
            "param_cb": lambda m: m.param_cb(m.parameter_names[-1]),
        },
        refused={
            "param_cb(lr)": lambda m: m.param_cb(
                m.parameter_names[-1], method="lr"
            ),
            "cb(bootstrap)": lambda m: m.cb(
                QUERY, np.zeros((3, 2)), method="bootstrap", random_state=1
            ),
        },
        covariates=True,
        rtol=5e-3,
    )


WEIGHTED = [
    _np("KaplanMeier"),
    _np("NelsonAalen"),
    _np("FlemingHarrington"),
    _parametric("Weibull"),
    _parametric("LogNormal"),
    _parametric("Exponential"),
    _parametric("Gamma"),
    _parametric("LogLogistic"),
    _regression("WeibullPH"),
    _regression("WeibullAFT"),
    _regression("LogNormalAFT"),
    _regression("WeibullPO"),
]
IDS = [case.name for case in WEIGHTED]


@pytest.fixture(scope="module")
def data():
    return weighted_reference_data()


def _fit(case, data, w, n=None, order=None):
    x, c, Z, tl = data["x"], data["c"], data["Z"], data["tl"]
    if order is not None:
        x, c, Z, tl = x[order], c[order], Z[order], tl[order]
        w = w[order]
        n = None if n is None else n[order]
    return quietly(case.fit, x, c, n, w, Z, tl)


def _bounds(case, model):
    return {
        k: np.asarray(f(model), dtype=float) for k, f in case.offered.items()
    }


@pytest.mark.parametrize("case", WEIGHTED, ids=IDS)
def test_unit_weights_give_the_unweighted_estimates(case, data):
    ones = np.ones(len(data["x"]))
    weighted = _fit(case, data, ones)
    plain = quietly(
        case.fit, data["x"], data["c"], None, None, data["Z"], data["tl"]
    )
    assert weighted.weighted and not getattr(plain, "weighted", False)
    assert_allclose(
        case.estimates(weighted), case.estimates(plain), rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize("scale", [0.1, 10.0])
@pytest.mark.parametrize("case", WEIGHTED, ids=IDS)
def test_weight_scale_changes_nothing(case, data, scale):
    base = _fit(case, data, data["w"])
    scaled = _fit(case, data, data["w"] * scale)
    assert_allclose(
        case.estimates(scaled), case.estimates(base), rtol=1e-5, atol=1e-8
    )
    for key, value in _bounds(case, base).items():
        assert_allclose(
            _bounds(case, scaled)[key],
            value,
            rtol=case.rtol,
            atol=1e-8,
            err_msg=key,
        )
    if hasattr(base, "covariance"):
        assert_allclose(
            scaled.covariance(), base.covariance(), rtol=case.rtol, atol=1e-10
        )


@pytest.mark.parametrize("case", WEIGHTED, ids=IDS)
def test_row_order_does_not_matter(case, data):
    base = _fit(case, data, data["w"])
    order = np.random.default_rng(3).permutation(len(data["x"]))
    permuted = _fit(case, data, data["w"], order=order)
    assert_allclose(
        case.estimates(permuted), case.estimates(base), rtol=1e-5, atol=1e-8
    )
    for key, value in _bounds(case, base).items():
        assert_allclose(
            _bounds(case, permuted)[key], value, rtol=case.rtol, atol=1e-8
        )


@pytest.mark.parametrize("case", WEIGHTED, ids=IDS)
def test_counts_still_equal_repetition(case, data):
    # The first five rows counted twice, against those rows written twice
    k = len(data["x"])
    n = np.ones(k, dtype=int)
    n[:5] = 2
    counted = _fit(case, data, data["w"], n=n)
    rows = np.concatenate([np.arange(k), np.arange(5)])
    repeated = {key: np.asarray(v)[rows] for key, v in data.items()}
    written = quietly(
        case.fit,
        repeated["x"],
        repeated["c"],
        None,
        repeated["w"],
        repeated["Z"],
        repeated["tl"],
    )
    assert_allclose(
        case.estimates(counted), case.estimates(written), rtol=1e-5, atol=1e-8
    )
    for key, value in _bounds(case, written).items():
        assert_allclose(_bounds(case, counted)[key], value, rtol=case.rtol)


@pytest.mark.parametrize("case", WEIGHTED, ids=IDS)
def test_offered_bounds_are_finite_and_refused_ones_refuse(case, data):
    model = _fit(case, data, data["w"])
    for key, value in _bounds(case, model).items():
        assert np.isfinite(value).all(), key
    for key, call in case.refused.items():
        with pytest.raises(WeightedRefusal, match="weights") as info:
            call(model)
        # Every refusal says what to use instead
        assert any(
            word in str(info.value) for word in ("Use", "use", "Fit")
        ), key
    for criterion in ("aic", "bic", "aic_c"):
        if not hasattr(model, criterion):
            continue  # a non-parametric estimate has none
        with pytest.raises(WeightedRefusal, match="pseudo-likelihood"):
            getattr(model, criterion)()


@pytest.mark.parametrize("case", WEIGHTED, ids=IDS)
def test_round_trip_keeps_the_robust_bounds(case, data):
    model = _fit(case, data, data["w"])
    d = model.to_dict()
    assert d["schema"] == 3 and d["weighted"] is True
    restored = sp.from_dict(json.loads(json.dumps(d, allow_nan=False)))
    pickled = pickle.loads(pickle.dumps(model))
    for again in (restored, pickled):
        assert again.weighted
        for key, call in case.offered.items():
            if key == "rmst" and again is restored:
                # The jackknife of the area needs the data
                with pytest.raises(ValueError, match="with_data=True"):
                    call(again)
                continue
            assert_allclose(call(again), call(model), rtol=1e-10)
        if hasattr(model, "covariance"):
            assert_allclose(again.covariance(), model.covariance())
        if hasattr(model, "aic"):
            with pytest.raises(WeightedRefusal):
                again.aic()


def test_refused_by_the_fitters_that_take_no_w(data):
    weighted = sp.SurpyvalData(data["x"], data["c"], w=data["w"])
    for fitter in (sp.Turnbull, sp.MixtureModel(sp.Weibull, 2)):
        with pytest.raises(ValueError, match="weights|w"):
            fitter.fit(weighted)
    with pytest.raises(WeightedRefusal, match="fit_best"):
        sp.fit_best(weighted)
    with pytest.raises(WeightedRefusal, match="how='MPP'"):
        sp.Weibull.fit(data["x"], data["c"], w=data["w"], how="MPP")


#: Fitters that take ``w`` only to refuse it (a later stage), with why.
REFUSES_W = {
    "Turnbull": "its weighted EM and variance are not built yet",
}


def test_turnbull_refuses_weights(data):
    with pytest.raises(WeightedRefusal, match="Turnbull"):
        sp.Turnbull.fit(data["x"], data["c"], w=data["w"])


def _takes_w(obj):
    try:
        return "w" in inspect.signature(obj.fit).parameters
    except (AttributeError, TypeError, ValueError):
        return False


def test_every_fitter_taking_w_is_registered():
    """Every public fitter whose ``fit`` takes ``w`` is in ``WEIGHTED``,
    or shares the fit of one that is: a parametric distribution (the
    ``OptimisedFitMixin`` fit and its sandwich) or a parametric regression
    (``fit_log_linear``, ``prepare_regression_fit`` and the regression
    model's sandwich), or is in ``REFUSES_W``."""
    from surpyval.univariate.parametric.optimised_fit import (
        OptimisedFitMixin,
    )
    from surpyval.univariate.regression.regression_data import (
        DataFrameRegressionMixin,
    )

    shared = (OptimisedFitMixin, DataFrameRegressionMixin)
    unregistered = []
    for name in sorted(dir(sp)):
        obj = getattr(sp, name)
        if not _takes_w(obj) or name in IDS or name in REFUSES_W:
            continue
        cls = obj if inspect.isclass(obj) else type(obj)
        if issubclass(cls, shared):
            continue
        unregistered.append(name)
    assert (
        not unregistered
    ), f"These fitters take w but are not registered: {unregistered}"
