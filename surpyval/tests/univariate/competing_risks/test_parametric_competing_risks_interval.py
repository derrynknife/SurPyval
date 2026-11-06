"""Left- and interval-censored rows in ``ParametricCompetingRisks``.

A unit found failed from cause ``k`` in ``(a, b]`` contributes
``CIF_k(b) - CIF_k(a)`` (over ``S(tl)`` with entry at ``tl``), which ties
the causes together, so the causes are fitted jointly. References:

- R 4.6.2 (``scripts/reference/
  reference_r_parametric_competing_risks_interval.R``, whose printed values
  are typed in below): the same likelihood written out in base R, its
  integrals by ``integrate()`` and its maximum by ``optim``, with
  numDeriv's Hessian for the covariance. No R package fits this model
  (flexsurv, cmprsk and mstate take exact times; icenReg has no competing
  risks);
- the probability of a window against :meth:`cif` (adaptive quadrature),
  its additivity over a split window, and its limit at a point (the
  sub-density, ``iif``);
- a single cause, where the model is the univariate fit to the same
  interval-censored data;
- the joint likelihood at the cause-by-cause estimate of data without
  windows, where it is the sum of the causes' likelihoods.
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import surpyval as sp
from surpyval import Weibull
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    ParametricCompetingRisks,
)
from surpyval.univariate.competing_risks.parametric import (
    _interval_likelihood as joint,
)

X = np.array(
    [4.74, 1.3, 8.49, 3.92, 0.21, 3.05, 3.64, 4.46, 1.57, 16.35, 0.13,
     10.95, 2.94, 4.03, 4.13, 6.08, 10.66, 4.59, 0.53, 0.95, 6.69, 4.62,
     0.88, 7.28, 2.93, 0.07, 12.74, 0.14, 4.81, 3.84, 4.74, 3.39, 19.74,
     5.85, 4.18, 5.85, 2.86, 7.08, 14.88, 15.95]
)  # fmt: skip
CAUSE = np.array(
    [0, 2, 1, 0, 2, 0, 1, 1, 2, 0, 0, 1, 1, 2, 1, 1, 1, 2, 0, 2, 2, 2, 1,
     1, 0, 2, 1, 2, 1, 2, 0, 0, 1, 0, 0, 2, 2, 1, 1, 1]
)  # fmt: skip
E = np.array([None if c == 0 else int(c) for c in CAUSE], dtype=object)


def _inspected():
    """The R reference's data: every other unit (from the first) found
    failed at an inspection every 2 time units, in ``(2j, 2j + 2]``
    (left censored at 2 for the first); the rest as observed."""
    insp = (np.arange(X.size) % 2 == 0) & (CAUSE != 0)
    hi = 2 * np.ceil(X / 2)
    lo = hi - 2
    x = np.column_stack(
        [np.where(insp, np.where(lo == 0, hi, lo), X), np.where(insp, hi, X)]
    )
    c = np.where(CAUSE == 0, 1, np.where(insp, np.where(lo == 0, -1, 2), 0))
    return x, c


# R 4.6.2: the joint maximum-likelihood fit (see the module docstring).
R_PARAMS = np.array([11.55636769, 1.996203052, 27.40924665, 0.6489515089])
R_NEG_LL = 91.2826406541
R_COV = np.array(
    [
        [2.1071492, -0.015818773, 0.019935895, -3.2035268e-05],
        [-0.015818773, 0.12591987, 0.0069100829, 0.00013474312],
        [0.019935895, 0.0069100829, 245.01301, -1.7832559],
        [-3.2035268e-05, 0.00013474312, -1.7832559, 0.029539203],
    ]
)
R_CIF = {
    1: np.array([0.09158746087, 0.2757064604, 0.5244156922]),
    2: np.array([0.2428291086, 0.329584336, 0.3811016568]),
}


@pytest.fixture(scope="module")
def model():
    x, c = _inspected()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return ParametricCompetingRisks.fit(x, E, c=c, dist=Weibull)


def _brute_neg_ll(model, x, c, e, tl=None):
    """The negative log-likelihood of the data at the model, from its own
    ``cif`` (adaptive quadrature), ``iif`` and ``sf``: an independent
    computation of every row's contribution."""
    xl, xr = (x[:, 0], x[:, 1]) if np.ndim(x) == 2 else (x, x)
    tl = np.full(xl.size, -np.inf) if tl is None else np.asarray(tl, float)
    ll = 0.0
    for i in range(xl.size):
        if c[i] == 1:
            ll += np.log(model.sf(xl[i]))
        elif c[i] == 0:
            ll += np.log(model.iif(xl[i], e[i]))
        else:
            a = tl[i] if c[i] == -1 else xl[i]
            low = model.cif(a, e[i]) if np.isfinite(a) else 0.0
            ll += np.log(model.cif(xr[i], e[i]) - low)
        if np.isfinite(tl[i]):
            ll -= np.log(model.sf(tl[i]))
    return -ll


# ---------------------------------------------------------------------------
# Against R
# ---------------------------------------------------------------------------
def test_matches_r_estimates_and_likelihood(model):
    # The likelihood is flat along cause 2's scale (standard error 16 on
    # 27), so optim's answer is only as close as its tolerance; the two
    # agree to 5e-8.
    np.testing.assert_allclose(model.params, R_PARAMS, rtol=1e-5)
    assert model.neg_ll() == pytest.approx(R_NEG_LL, abs=1e-7)
    assert model.maximum == "verified"


def test_matches_r_covariance(model):
    # numDeriv's Richardson Hessian of integrate()'s likelihood against
    # autograd's of the quadrature: agreement to 1e-3 relative.
    np.testing.assert_allclose(model.covariance(), R_COV, rtol=1e-3)
    # The windows tie the causes: the terms between them are not 0.
    assert abs(model.covariance()[0, 2]) > 1e-3


def test_matches_r_cumulative_incidence(model):
    for k in (1, 2):
        np.testing.assert_allclose(
            model.cif([4.0, 8.0, 15.0], k), R_CIF[k], rtol=2e-6
        )


def test_likelihood_is_every_rows_contribution(model):
    x, c = _inspected()
    assert model.neg_ll() == pytest.approx(
        _brute_neg_ll(model, x, c, E), rel=1e-9
    )


# ---------------------------------------------------------------------------
# The window probability
# ---------------------------------------------------------------------------
SPECS = [
    {"a": (sp.Weibull, [10.0, 2.0]), "b": (sp.Weibull, [15.0, 0.5])},
    {"a": (sp.LogNormal, [2.0, 0.2]), "b": (sp.Gamma, [0.5, 0.05])},
    {"a": (sp.Normal, [10.0, 3.0]), "b": (sp.Weibull, [12.0, 2.0])},
    {
        "a": (sp.Weibull, [10.0, 6.0]),
        "b": (sp.Weibull, [15.0, 0.3]),
        "c": (sp.LogLogistic, [20.0, 3.0]),
    },
]
WINDOWS = [(-np.inf, 5.0), (-np.inf, 1000.0), (2.0, 5.0), (1e-6, 50.0)]
WINDOWS += [(0.001, 0.002), (9.0, 1e4), (12.0, 12.01)]


def _probabilities(spec, cause, windows):
    """``joint.window_probability`` of ``windows`` of ``cause``, with the
    quadrature set at the truth, in the order given."""
    causes = sorted(spec)
    dists = [spec[k][0] for k in causes]
    theta = np.concatenate([spec[k][1] for k in causes])
    out = []
    for a, b in windows:
        left = not np.isfinite(a)
        x = np.array([[b, b] if left else [a, b]])
        data = joint.prepare(
            causes,
            dists,
            x,
            np.array([-1 if left else 2]),
            np.ones(1),
            np.array([cause], dtype=object),
            np.array([[-np.inf, np.inf]]),
        )
        data.adapt(theta)
        i = causes.index(cause)
        params = joint.split(theta, data.sizes)
        out.append(
            float(joint.window_probability(i, data.rows[i], data, params)[0])
        )
    return np.array(out)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: "-".join(sorted(s)))
def test_window_probability_is_the_cif_difference(spec):
    model = ParametricCompetingRisks.from_fitted(
        {k: d.from_params(p) for k, (d, p) in spec.items()}
    )
    for k in spec:
        got = _probabilities(spec, k, WINDOWS)
        ref = np.array(
            [
                model.cif(b, k) - (model.cif(a, k) if np.isfinite(a) else 0)
                for a, b in WINDOWS
            ]
        )
        # cif holds to about 1e-10; this quadrature to about 1e-12.
        np.testing.assert_allclose(got, ref, rtol=1e-8)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: "-".join(sorted(s)))
def test_window_probability_adds_over_a_split(spec):
    # P(a, c] = P(a, b] + P(b, c], and a left-censored row from the start
    # is the window (start, b] split anywhere.
    for k in spec:
        whole, first, second = _probabilities(
            spec, k, [(2.0, 9.0), (2.0, 4.5), (4.5, 9.0)]
        )
        assert whole == pytest.approx(first + second, rel=1e-10)
        left, below, above = _probabilities(
            spec, k, [(-np.inf, 9.0), (-np.inf, 3.0), (3.0, 9.0)]
        )
        assert left == pytest.approx(below + above, rel=1e-10)


def test_window_probability_at_a_point_is_the_sub_density():
    # An exact failure at t is the limit of the window (t - eps, t]
    # scaled by 1 / eps: the sub-density h_k(t) S(t), iif.
    spec = SPECS[0]
    model = ParametricCompetingRisks.from_fitted(
        {k: d.from_params(p) for k, (d, p) in spec.items()}
    )
    for k in spec:
        for t in (0.5, 6.0, 20.0):
            eps = 1e-7 * t
            p = _probabilities(spec, k, [(t - eps, t)])[0]
            assert p / eps == pytest.approx(model.iif(t, k), rel=1e-6)


# ---------------------------------------------------------------------------
# Against the univariate fit and the cause-by-cause fit
# ---------------------------------------------------------------------------
def test_one_cause_is_the_univariate_interval_fit():
    x, c = _inspected()
    e = np.where(CAUSE == 0, None, "fail").astype(object)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ParametricCompetingRisks.fit(x, e, c=c, dist=Weibull)
        uni = Weibull.fit(x=x, c=c)
    np.testing.assert_allclose(model.params, uni.params, rtol=1e-6)
    assert model.neg_ll() == pytest.approx(uni.neg_ll(), abs=1e-8)
    np.testing.assert_allclose(model.covariance(), uni.covariance(), rtol=1e-4)
    assert model.bic() == pytest.approx(uni.bic(), abs=1e-6)


def test_without_windows_the_joint_likelihood_is_the_causes_sum():
    # The joint likelihood's exact, censored and entry terms, at the
    # cause-by-cause estimate of data without windows.
    tl = np.where((np.arange(X.size) % 4 == 0) & (X > 1), 0.5, 0.0)
    model = ParametricCompetingRisks.fit(X, E, tl=tl, dist=Weibull)
    assert model._joint is None
    c = np.where(CAUSE == 0, 1, 0)
    t = np.column_stack([tl, np.full(X.size, np.inf)])
    data = joint.prepare(
        [1, 2], [Weibull, Weibull], X, c, np.ones(X.size), E, t
    )
    assert float(joint.neg_ll(model.params, data)) == pytest.approx(
        model.neg_ll(), rel=1e-12
    )


def test_without_windows_the_fit_is_cause_by_cause():
    model = ParametricCompetingRisks.fit(X, E, dist=Weibull)
    assert model._joint is None
    for k in (1, 2):
        own = Weibull.fit(X, c=np.where(CAUSE == k, 0, 1))
        np.testing.assert_array_equal(model.models[k].params, own.params)
    assert "joint" not in model.to_dict()


def test_left_censored_rows_with_entry():
    # A unit entering at tl and found failed at x contributes
    # (CIF_k(x) - CIF_k(tl)) / S(tl).
    x, c = _inspected()
    tl = np.where((np.arange(X.size) % 3 == 0) & (x[:, 0] > 1), 0.5, 0.0)
    tl = np.where(c == -1, 0.5, tl)
    model = ParametricCompetingRisks.fit(x, E, c=c, tl=tl, dist=Weibull)
    assert model.maximum == "verified"
    assert model.neg_ll() == pytest.approx(
        _brute_neg_ll(model, x, c, E, tl), rel=1e-9
    )


# ---------------------------------------------------------------------------
# Every bound and the comparison values
# ---------------------------------------------------------------------------
def test_bounds_contain_the_estimate(model):
    t = np.array([2.0, 6.0, 15.0])
    for k in (1, 2):
        band = model.cb(t, k)
        F = model.cif(t, k)
        assert np.all((band[:, 0] < F) & (F < band[:, 1]))
        net = model.cb(t, k, on="sf")
        S = model.sf(t, event=k)
        assert np.all((net[:, 0] < S) & (S < net[:, 1]))
    for on, f in (("sf", model.sf), ("ff", model.ff), ("Hf", model.Hf)):
        band = model.cb(t, on=on)
        assert np.all((band[:, 0] < f(t)) & (f(t) < band[:, 1]))


def test_all_cause_bound_carries_the_terms_between_causes(model):
    # The bound on log H by the delta method with the full covariance and
    # the Weibull cumulative hazard's analytic gradient.
    t = np.array([3.0, 8.0])
    a1, b1, a2, b2 = model.params
    grad = []
    for alpha, beta in ((a1, b1), (a2, b2)):
        z = (t / alpha) ** beta
        grad += [-beta / alpha * z, z * np.log(t / alpha)]
    grad = np.array(grad)
    H = model.Hf(t)
    sd = np.sqrt(np.einsum("it,ij,jt->t", grad, model.covariance(), grad)) / H
    z95 = norm.ppf(0.975)
    expected = np.column_stack([H * np.exp(-z95 * sd), H * np.exp(z95 * sd)])
    np.testing.assert_allclose(model.cb(t, on="Hf"), expected, rtol=1e-6)


def test_net_bound_is_the_cause_block(model):
    # A cause's net function depends on its parameters alone: its Wald
    # bound uses its block of the joint covariance.
    np.testing.assert_array_equal(
        model.models[1].covariance(), model.covariance()[:2, :2]
    )
    with pytest.raises(ValueError, match="use method='wald'"):
        model.cb(5.0, 1, on="sf", method="lr")
    with pytest.raises(ValueError, match="use method='wald'"):
        model.cb(5.0, 1, on="sf", method="bootstrap")


def test_standard_errors_and_information_criteria(model):
    np.testing.assert_allclose(
        model.standard_errors(), np.sqrt(np.diag(model.covariance()))
    )
    failures = np.sum(CAUSE != 0)
    assert model.aic() == pytest.approx(2 * 4 + 2 * model.neg_ll())
    assert model.bic() == pytest.approx(
        2 * model.neg_ll() + 4 * np.log(failures)
    )
    assert model.log_likelihood == -model.neg_ll()


def test_with_params_is_a_model_without_a_fit(model):
    other = model.with_params(model.params)
    assert other._joint is None
    np.testing.assert_allclose(
        other.cif([4.0, 8.0], 1), model.cif([4.0, 8.0], 1)
    )


def test_serialisation_keeps_the_joint_fit(model):
    d = json.loads(json.dumps(model.to_dict()))
    back = ParametricCompetingRisks.from_dict(d)
    np.testing.assert_array_equal(back.covariance(), model.covariance())
    assert back.neg_ll() == model.neg_ll()
    assert back.bic() == model.bic()
    np.testing.assert_array_equal(
        back.cb([4.0, 8.0], 2), model.cb([4.0, 8.0], 2)
    )
    again = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(again.covariance(), model.covariance())


def test_fit_from_df_with_windows(model):
    x, c = _inspected()
    df = pd.DataFrame({"lo": x[:, 0], "hi": x[:, 1], "e": E, "c": c})
    from_df = ParametricCompetingRisks.fit_from_df(
        df, None, "e", c_col="c", xl_col="lo", xr_col="hi"
    )
    np.testing.assert_allclose(from_df.params, model.params, rtol=1e-9)


def test_flags_derived_from_the_windows(model):
    # Without c, a row [xl, xr] with xl < xr is interval censored, one
    # with no lower end left censored, as every fit derives them.
    x, c = _inspected()
    x = np.where(
        (c == -1)[:, None], np.column_stack([-np.inf * x[:, 0], x[:, 1]]), x
    )
    derived = ParametricCompetingRisks.fit(x, E, dist=Weibull)
    np.testing.assert_allclose(derived.params, model.params, rtol=1e-9)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_refuses_a_window_without_a_cause():
    with pytest.raises(ValueError, match="needs its cause"):
        ParametricCompetingRisks.fit(
            [[1.0, 1.0], [2.0, 3.0], [4.0, 4.0]],
            [1, None, 2],
            c=[0, 2, 0],
        )


def test_refuses_other_estimation_methods():
    x, c = _inspected()
    with pytest.raises(ValueError, match="use how='MLE'"):
        ParametricCompetingRisks.fit(x, E, c=c, how="MPS")


def test_refuses_a_discrete_distribution():
    x, c = _inspected()
    with pytest.raises(ValueError, match="continuous distributions only"):
        ParametricCompetingRisks.fit(x, E, c=c, dist=sp.Geometric)


def test_refuses_an_infinite_upper_end():
    # [xl, inf] is a unit right censored at xl, with no cause.
    with pytest.raises(ValueError, match="right censored at xl"):
        ParametricCompetingRisks.fit(
            [[1.0, 1.0], [2.0, np.inf], [4.0, 4.0]], [1, 1, 2], c=[0, 2, 0]
        )


def test_non_parametric_refusal_names_the_parametric_model():
    with pytest.raises(ValueError, match="ParametricCompetingRisks"):
        CompetingRisks.fit([1.0, 2.0, 3.0], [1, 2, 1], c=[0, -1, 0])


def test_joint_derivatives_agree_with_finite_differences(model):
    # autograd's gradient and Hessian of the joint likelihood (with the
    # quadrature held as the fit left it) against central differences.
    from autograd import grad, hessian

    x, c = _inspected()
    t = np.column_stack([np.full(X.size, -np.inf), np.full(X.size, np.inf)])
    data = joint.prepare(
        [1, 2], [Weibull, Weibull], x, c, np.ones(X.size), E, t
    )
    theta = model.params
    data.adapt(theta)
    f = lambda v: joint.neg_ll(v, data)  # noqa: E731
    h = 1e-5 * np.abs(theta)
    step = np.diag(h)
    fd_grad = np.array(
        [(f(theta + s) - f(theta - s)) / (2 * hj) for s, hj in zip(step, h)]
    )
    g = grad(f)
    fd_hess = np.array(
        [(g(theta + s) - g(theta - s)) / (2 * hj) for s, hj in zip(step, h)]
    )
    np.testing.assert_allclose(grad(f)(theta), fd_grad, atol=1e-6)
    np.testing.assert_allclose(
        hessian(f)(theta), fd_hess, rtol=1e-5, atol=1e-8
    )
