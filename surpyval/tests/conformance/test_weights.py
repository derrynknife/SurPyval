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
    #: test name -> reason: a property this case is known to fail (a
    #: suspected bug, kept as a strict xfail until it is fixed)
    xfail: dict = field(default_factory=dict)


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


def _parametric(dist, no_tl=False, discrete=False):
    def fit(x, c, n, w, Z, tl):
        if discrete:
            # Whole numbers: the times rounded up, the entries down
            x, tl = np.ceil(x), np.floor(tl)
        return getattr(sp, dist).fit(x, c, n, w=w, tl=None if no_tl else tl)

    return WeightedCase(
        dist,
        fit,
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


def _regression(fitter, no_tl=False):
    return WeightedCase(
        fitter,
        lambda x, c, n, w, Z, tl: getattr(sp, fitter).fit(
            x, Z, c=c, n=n, w=w, tl=None if no_tl else tl
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
    # audit W-A: the shapes of distribution the five above do not cover.
    # The Normal and Gumbel fits are given no left truncation: with the
    # fixture's late entries their (weighted or not) maximum runs off to
    # mu -> -inf (the truncated Normal's exponential limit; the Gumbel's
    # search does not converge), so there is no estimate to compare.
    _parametric("Normal", no_tl=True),
    _parametric("Gumbel", no_tl=True),
    _parametric("Rayleigh"),  # one parameter
    # Discrete: whole-number data (the times rounded up, the entries down)
    _parametric("Poisson", discrete=True),
    # audit W-A: additive hazards (principle 9). Its fit to the fixture's
    # late entries sits on the boundary beta = 1, where the covariance is
    # NaN (weighted or not), so it is fitted without the truncation; there
    # cb(sf) at Z = 0 is finite, and is offered.
    _regression("WeibullAH", no_tl=True),
    # audit W-A: an accelerated-life model (a Weibull life at a power-law
    # stress); the stress is 1 + the first covariate, so positive.
    WeightedCase(
        "AcceleratedLife[Weibull,Power]",
        lambda x, c, n, w, Z, tl: sp.AcceleratedLife(
            sp.Weibull, sp.life_models.Power
        ).fit(x, 1 + Z[:, 0], c=c, n=n, w=w, tl=tl),
        lambda m: np.asarray(m.params, dtype=float),
        offered={
            "cb(sf)": lambda m: m.cb(QUERY, 1.5, on="sf"),
            "param_cb": lambda m: m.param_cb(m.parameter_names[-1]),
        },
        refused={
            "param_cb(lr)": lambda m: m.param_cb(
                m.parameter_names[-1], method="lr"
            ),
            "cb(bootstrap)": lambda m: m.cb(
                QUERY, 1.5, method="bootstrap", random_state=1
            ),
        },
        covariates=True,
        rtol=5e-3,
        xfail={
            name: (
                "AcceleratedLife's fit stops after one TNC iteration on "
                "'|f_n-f_(n-1)| ~= 0' with a gradient of about 2e-4 left, "
                "so fits that reach the same maximum (w = 1 against no w; "
                "counted against repeated rows) differ by 1e-5 to 4e-5 "
                "relative in a and n. Reproduce: AcceleratedLife(Weibull, "
                "Power).fit(x, 1 + Z[:, 0], c=c, tl=tl) against the same "
                "with w=np.ones(40) on weighted_reference_data(): n is "
                "-1.355954 against -1.355935."
            )
            for name in ("unit_weights", "counts", "integer_weights")
        },
    ),
    # audit W-B: the forms of the parametric fit with their own branches
    # in the sandwich (``make_robust``): a limited failure population, zero
    # inflation, an offset and held parameters.
    WeightedCase(
        "Weibull[lfp]",
        lambda x, c, n, w, Z, tl: sp.Weibull.fit(
            np.minimum(x, 8.0),
            np.where(x > 8.0, 1, c),
            n,
            w=w,
            tl=tl,
            lfp=True,
        ),
        lambda m: np.array([*m.params, m.lfp_p], dtype=float),
        offered={
            "cb(sf)": lambda m: m.cb(QUERY, on="sf"),
            "param_cb": lambda m: m.param_cb("alpha"),
            "quantile_cb": lambda m: m.quantile_cb(0.3),
        },
    ),
    # The exact zeros are the failures at or before 1.0 (four rows), chosen
    # by value so that reordering the rows moves them too. No left
    # truncation: a zero cannot be observed after an entry time (not even
    # one of 0), as ``SurpyvalData`` requires.
    WeightedCase(
        "Weibull[zi]",
        lambda x, c, n, w, Z, tl: sp.Weibull.fit(
            np.where(x <= 1.0, 0.0, x), c, n, w=w, zi=True
        ),
        lambda m: np.array([*m.params, m.f0], dtype=float),
        offered={
            "cb(sf)": lambda m: m.cb(QUERY, on="sf"),
            "param_cb": lambda m: m.param_cb("alpha"),
            "quantile_cb": lambda m: m.quantile_cb(0.3),
        },
    ),
    # The data and the entry times shifted by 3: the bounds are asked
    # above the offset, where the survival is not 1.
    WeightedCase(
        "Weibull[offset]",
        lambda x, c, n, w, Z, tl: sp.Weibull.fit(
            x + 3, c, n, w=w, tl=tl + 3, offset=True
        ),
        lambda m: np.array([*m.params, m.gamma], dtype=float),
        # (each bound warns that it holds gamma at its estimate)
        offered={
            "cb(sf)": lambda m: quietly(m.cb, QUERY + 3, on="sf"),
            "param_cb": lambda m: quietly(m.param_cb, "alpha"),
            "quantile_cb": lambda m: quietly(m.quantile_cb, 0.3),
        },
    ),
    WeightedCase(
        "Weibull[fixed]",
        lambda x, c, n, w, Z, tl: sp.Weibull.fit(
            x, c, n, w=w, tl=tl, fixed={"beta": 2.0}
        ),
        lambda m: np.asarray(m.params, dtype=float),
        offered={
            "cb(sf)": lambda m: m.cb(QUERY, on="sf"),
            "param_cb": lambda m: m.param_cb("alpha"),
            "quantile_cb": lambda m: m.quantile_cb(0.3),
        },
    ),
    WeightedCase(
        "CoxPH",
        lambda x, c, n, w, Z, tl: sp.CoxPH.fit(x, Z, c=c, n=n, w=w, tl=tl),
        lambda m: np.asarray(m.params, dtype=float),
        refused={
            "compute_residuals": lambda m: m.compute_residuals(),
            "check_ph": lambda m: m.check_ph(),
        },
        covariates=True,
    ),
]
IDS = [case.name for case in WEIGHTED]


def _cases(test):
    """``WEIGHTED`` as parameters of ``test``, each case known to fail it
    marked a strict xfail with its reason."""
    return [
        pytest.param(
            case,
            id=case.name,
            marks=(
                [pytest.mark.xfail(strict=True, reason=case.xfail[test])]
                if test in case.xfail
                else []
            ),
        )
        for case in WEIGHTED
    ]


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


@pytest.mark.parametrize("case", _cases("unit_weights"))
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
@pytest.mark.parametrize("case", _cases("scale"))
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


@pytest.mark.parametrize("case", _cases("order"))
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


@pytest.mark.parametrize("case", _cases("counts"))
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


def _value(attr):
    """An attribute that is a number on some models and a method on
    others."""
    return attr() if callable(attr) else attr


def _breslow_cox(x, c, n, w, Z, tl):
    return sp.CoxPH.fit(x, Z, c=c, n=n, w=w, tl=tl, tie_method="breslow")


#: The cases whose fit with whole-number weights is not expected to equal
#: the counted fit, with why.
NOT_COUNTED = {
    "FlemingHarrington": (
        "by design: each of the m tied deaths is taken at the deaths' mean "
        "weight, m the units from the counts (R's ctype = 2; docs/Weighted "
        "Data.rst), so a death of weight 2 is one death, not two tied ones"
    ),
    "WeibullAH": (
        "the fit ends on the boundary of the model's support (it warns), "
        "not at a maximum, with or without weights"
    ),
    "Weibull[offset]": (
        "with these weights (or these counts) the likelihood has no finite "
        "maximum: gamma runs onto the smallest observation with beta < 1, "
        "and the fit warns 'No finite maximum'"
    ),
}


@pytest.mark.parametrize("case", _cases("integer_weights"))
def test_integer_weights_give_the_counted_estimates(case, data):
    """audit W-J: whole-number weights give the point estimates of the
    same rows counted that many times (``n = w``, no ``w``): a check of the
    weighted likelihoods and baselines that ``w = 1`` cannot make. Only
    the estimates agree; the intervals differ by design (robust against
    classical). A Cox fit is compared under Breslow's ties, since under
    Efron's R's weighted convention differs from repeated rows (DECISIONS.md
    "Ties in a weighted Cox fit"), and its baseline is compared too.

    The log-likelihood (reported with the weights as given) must equal the
    counted one to 1e-9; the estimates are compared to 1e-5, the tolerance
    of the other properties between fits that reach the same maximum: the
    optimisers stop within about 3e-6 of it (the Exponential, LogNormalAFT
    and Weibull[lfp] estimates differ by 1e-6 to 3e-6 while their
    log-likelihoods agree to 1e-12)."""
    if case.name in NOT_COUNTED:
        pytest.skip(NOT_COUNTED[case.name])
    k = len(data["x"])
    w = np.where(np.arange(k) % 2 == 0, 2, 1)
    fit = _breslow_cox if case.name == "CoxPH" else case.fit
    args = (data["x"], data["c"])
    weighted = quietly(
        fit, *args, None, w.astype(float), data["Z"], data["tl"]
    )
    counted = quietly(fit, *args, w, None, data["Z"], data["tl"])
    assert weighted.weighted and not getattr(counted, "weighted", False)
    assert_allclose(
        case.estimates(weighted), case.estimates(counted), rtol=1e-5, atol=1e-9
    )
    if hasattr(counted, "log_likelihood"):
        assert_allclose(
            _value(weighted.log_likelihood),
            _value(counted.log_likelihood),
            rtol=1e-9,
        )
    if case.name == "CoxPH":
        zero = np.zeros((3, 2))
        assert_allclose(
            weighted.sf(QUERY, zero), counted.sf(QUERY, zero), rtol=1e-6
        )


@pytest.mark.parametrize("case", _cases("bounds"))
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


@pytest.mark.parametrize("case", _cases("round_trip"))
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
