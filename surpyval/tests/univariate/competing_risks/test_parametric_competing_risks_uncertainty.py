"""Parameter uncertainty of a parametric competing-risks model:
``params``, ``parameter_names``, ``covariance_names``, ``covariance``,
``standard_errors``, ``with_params`` and the Wald bounds of ``cb``.

References:

- R flexsurv 2.3.2 (``scripts/reference/
  reference_r_parametric_competing_risks.R``, whose printed values are
  typed in below): each cause's Weibull fit with the other cause
  censored, ``flexsurvreg``, its estimates and covariance;
  the cumulative incidence and all-cause survival of the competing-risks
  multi-state model built from the two fits (``fmsm``, ``pmatrix.fs``),
  and its simulated intervals (compared loosely: they are percentiles of
  simulated parameters, not the delta method);
- closed forms: exponential causes, whose cumulative incidence
  :math:`\\lambda_k / \\Lambda\\,(1 - e^{-\\Lambda t})` has an analytic
  gradient, and a single cause, whose all-cause bound is the cause's own;
- the conformance contract of ``with_params`` (``with_params(_estimates())``
  reproduces every prediction; a wrong length raises ``ValueError``).
"""

import json
import warnings

import numpy as np
import pytest
from scipy.special import logit
from scipy.stats import norm

import surpyval
from surpyval import Exponential, Weibull
from surpyval.univariate.competing_risks import ParametricCompetingRisks

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
TIMES = np.array([4.0, 8.0, 15.0])
Q = np.array([0.5, 2.0, 4.0, 8.0, 15.0, 30.0])
FUNCTIONS = ("sf", "ff", "Hf", "hf", "df")


@pytest.fixture(scope="module")
def model():
    return ParametricCompetingRisks.fit(X, E, dist=Weibull)


def _extended():
    """Cause 1 a Weibull with a limited-failure proportion and zero
    inflation, cause 2 a Weibull with an offset, assembled with
    ``from_fitted``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = Weibull.fit(X, c=np.where(CAUSE == 1, 0, 1), lfp=True, zi=False)
        b = Weibull.fit(X + 2.0, c=np.where(CAUSE == 2, 0, 1), offset=True)
    return ParametricCompetingRisks.from_fitted({1: a, 2: b})


# ---------------------------------------------------------------------------
# Names, params and covariance
# ---------------------------------------------------------------------------
def test_names_and_params(model):
    assert model.parameter_names == [
        "1: alpha",
        "1: beta",
        "2: alpha",
        "2: beta",
    ]
    assert model.covariance_names == model.parameter_names
    np.testing.assert_array_equal(
        model.params, np.r_[model.models[1].params, model.models[2].params]
    )
    np.testing.assert_array_equal(model._estimates(), model.params)


def test_extended_names_follow_each_cause():
    m = _extended()
    assert m.covariance_names == [
        "1: alpha",
        "1: beta",
        "1: lfp_p",
        "2: alpha",
        "2: beta",
    ]
    assert len(m.parameter_names) == m.params.size == 4
    np.testing.assert_array_equal(
        m._estimates(),
        np.r_[m.models[1].params, m.models[1].lfp_p, m.models[2].params],
    )


@pytest.mark.parametrize("build", ["fit", "extended"])
def test_covariance_is_block_diagonal_of_the_causes(model, build):
    m = model if build == "fit" else _extended()
    cov = m.covariance()
    sizes = [len(m.models[k].covariance_names) for k in m.causes]
    assert cov.shape == (sum(sizes),) * 2
    start = 0
    for k, size in zip(m.causes, sizes):
        block = slice(start, start + size)
        np.testing.assert_array_equal(
            cov[block, block], m.models[k].covariance()
        )
        rest = np.ones(cov.shape[0], bool)
        rest[block] = False
        assert np.all(cov[block][:, rest] == 0)
        start += size
    np.testing.assert_array_equal(m.standard_errors(), np.sqrt(np.diag(cov)))


def test_no_covariance_names_the_cause():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = Weibull.fit(X, c=np.where(CAUSE == 1, 0, 1))
        b = Weibull.fit(X, c=np.where(CAUSE == 2, 0, 1), how="MPS")
    m = ParametricCompetingRisks.from_fitted({1: a, 2: b})
    for call in (
        m.covariance,
        m.standard_errors,
        lambda: m.cb(TIMES, 1),
        lambda: m.cb(TIMES, on="sf"),
    ):
        with pytest.raises(ValueError, match="cause 2 has none"):
            call()
    # cause 1's own bounds still work
    assert m.cb(TIMES, 1, on="sf").shape == (3, 2)


# ---------------------------------------------------------------------------
# with_params
# ---------------------------------------------------------------------------
def _predictions(m):
    out = {f: getattr(m, f)(Q) for f in FUNCTIONS}
    for k in m.causes:
        out[f"cif{k}"] = m.cif(Q, k)
        out[f"sf{k}"] = m.sf(Q, event=k)
    out["qf"] = m.qf(np.array([0.1, 0.3]))
    out["mean"] = m.mean(defective=True)
    return out


@pytest.mark.parametrize("build", ["fit", "extended"])
def test_with_params_of_estimates_reproduces_predictions(model, build):
    m = model if build == "fit" else _extended()
    before = _predictions(m)
    rebuilt = m.with_params(m._estimates())
    assert isinstance(rebuilt, ParametricCompetingRisks)
    assert rebuilt is not m and rebuilt.causes == m.causes
    after = _predictions(rebuilt)
    for key, value in before.items():
        np.testing.assert_allclose(
            after[key], value, rtol=1e-12, atol=0, err_msg=key
        )
    # The short form (distribution parameters only) keeps lfp_p
    short = m.with_params(m.params)
    for key, value in before.items():
        np.testing.assert_allclose(
            _predictions(short)[key], value, rtol=1e-12, err_msg=key
        )
    # Unchanged: the model, and its fitted offset
    np.testing.assert_array_equal(m._estimates(), rebuilt._estimates())
    for k in m.causes:
        assert rebuilt.models[k].extras == m.models[k].extras


def test_with_params_wrong_length_raises(model):
    with pytest.raises(ValueError, match="4 values.*1: alpha"):
        model.with_params([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="flat vector"):
        model.with_params(np.ones((2, 2)))
    m = _extended()
    with pytest.raises(ValueError, match="5 values.*or 4"):
        m.with_params(np.ones(3))


def test_with_params_moves_the_given_values_and_keeps_the_rest():
    m = _extended()
    values = m._estimates().copy()
    values[2] = 0.4  # cause 1's lfp_p
    values[3] *= 1.5  # cause 2's alpha
    other = m.with_params(values)
    assert other.models[1].lfp_p == 0.4
    assert other.models[2].params[0] == values[3]
    assert other.models[2].gamma == m.models[2].gamma
    moved = m.with_params(values, gamma={2: 1.0})
    assert moved.models[2].gamma == 1.0
    with pytest.raises(ValueError, match="cause 1 was not fitted with an"):
        m.with_params(values, gamma={1: 1.0})
    with pytest.raises(ValueError, match="lfp_p|between"):
        m.with_params(np.r_[values[:2], 1.5, values[3:]])
    # The original is untouched
    assert m.models[1].lfp_p != 0.4


def test_with_params_of_a_refit_matches_the_refit(model):
    other = ParametricCompetingRisks.fit(X[:30], E[:30], dist=Weibull)
    rebuilt = model.with_params(other._estimates())
    for key, value in _predictions(other).items():
        np.testing.assert_allclose(
            _predictions(rebuilt)[key], value, rtol=1e-12, err_msg=key
        )


def test_rebuilt_model_has_no_covariance(model):
    rebuilt = model.with_params(model._estimates())
    with pytest.raises(ValueError, match="cause 1 has none"):
        rebuilt.cb(TIMES, 1)


def test_covariance_survives_serialisation(model):
    saved = json.loads(json.dumps(model.to_dict()))
    restored = surpyval.from_dict(saved)
    np.testing.assert_allclose(
        restored.covariance(), model.covariance(), rtol=1e-12
    )
    np.testing.assert_allclose(
        restored.cb(TIMES, 1), model.cb(TIMES, 1), rtol=1e-9
    )


# ---------------------------------------------------------------------------
# cb
# ---------------------------------------------------------------------------
def test_single_cause_all_cause_bound_is_the_cause_bound():
    # With one cause the all-cause functions are the cause's, and a
    # Weibull's own bound is on log H too.
    c = np.where(CAUSE == 0, 1, 0)
    single = Weibull.fit(X, c=c)
    m = ParametricCompetingRisks.from_fitted({"a": single})
    for on in ("sf", "ff", "Hf"):
        for bound in ("two-sided", "lower", "upper"):
            np.testing.assert_allclose(
                m.cb(TIMES, on=on, bound=bound),
                single.cb(TIMES, on=on, bound=bound),
                rtol=1e-9,
                err_msg=(on, bound),
            )


def test_exponential_cif_bound_against_its_analytic_gradient():
    x = X
    e = np.where(CAUSE == 0, None, np.where(CAUSE == 1, "a", "b"))
    m = ParametricCompetingRisks.fit(x, e.astype(object), dist=Exponential)
    la, lb = m.params
    L = la + lb
    cov = m.covariance()
    for t in TIMES:
        F = la / L * -np.expm1(-L * t)
        # d/d la and d/d lb of la / L * (1 - exp(-L t))
        common = la / L * t * np.exp(-L * t) - la / L**2 * -np.expm1(-L * t)
        grad = np.array([common + -np.expm1(-L * t) / L, common])
        sd = np.sqrt(grad @ cov @ grad) / (F * (1 - F))
        zq = norm.ppf(0.975)
        expected = 1 / (1 + np.exp(-(logit(F) + np.array([-1, 1]) * zq * sd)))
        np.testing.assert_allclose(m.cb(t, "a"), expected, rtol=1e-6)


def test_all_cause_variance_is_the_sum_of_the_causes(model):
    from surpyval.univariate.competing_risks.parametric import (
        parametric_competing_risks as pcr,
    )

    for m in (model, _extended()):
        cov = m.covariance()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grad = pcr._central_differences(
                lambda v: m.with_params(v).Hf(TIMES),
                m._estimates(),
                cov,
                TIMES.size,
            )
            analytic = sum(m._var_Hf_cause(k, TIMES) for k in m.causes)
        numeric = np.einsum("jt,jk,kt->t", grad, cov, grad)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6)


@pytest.mark.parametrize(
    "on, event", [("cif", 1), ("cif", 2), ("sf", None), ("ff", None),
                  ("Hf", None), ("sf", 1), ("Hf", 2)]
)  # fmt: skip
def test_bounds_contain_the_estimate_and_nest(model, on, event):
    point = {
        "cif": lambda: model.cif(TIMES, event),
        "sf": lambda: model.sf(TIMES, event=event),
        "ff": lambda: model.ff(TIMES, event=event),
        "Hf": lambda: model.Hf(TIMES, event=event),
    }[on]()
    two = model.cb(TIMES, event, on=on)
    assert two.shape == (3, 2)
    assert np.all(two[:, 0] < point) and np.all(point < two[:, 1])
    wide = model.cb(TIMES, event, on=on, alpha_ci=0.01)
    assert np.all(wide[:, 0] < two[:, 0]) and np.all(two[:, 1] < wide[:, 1])
    lower = model.cb(TIMES, event, on=on, bound="lower", alpha_ci=0.1)
    upper = model.cb(TIMES, event, on=on, bound="upper", alpha_ci=0.1)
    ten = model.cb(TIMES, event, on=on, alpha_ci=0.2)
    np.testing.assert_allclose(lower, ten[:, 0], rtol=1e-6)
    np.testing.assert_allclose(upper, ten[:, 1], rtol=1e-6)
    if on != "Hf":
        assert np.all((two > 0) & (two < 1))


def test_cb_shapes_and_edges(model):
    assert model.cb(8.0, 1).shape == (2,)
    assert np.ndim(model.cb(8.0, 1, bound="lower")) == 0
    assert model.cb([[4.0, 8.0]], on="sf").shape == (1, 2, 2)
    assert model.cb([], 1).shape == (0, 2)
    out = model.cb([0.0, np.nan], 1)
    np.testing.assert_array_equal(out[0], [0.0, 0.0])
    assert np.all(np.isnan(out[1]))
    np.testing.assert_array_equal(model.cb(0.0, on="sf"), [1.0, 1.0])
    # sf and ff agree, from one bound on log H
    np.testing.assert_allclose(
        model.cb(TIMES, on="ff"), 1 - model.cb(TIMES, on="sf")[:, ::-1]
    )


def test_net_bounds_are_the_causes_own(model):
    for k in model.causes:
        for on in ("sf", "ff", "Hf"):
            np.testing.assert_array_equal(
                model.cb(TIMES, k, on=on), model.models[k].cb(TIMES, on=on)
            )
    np.testing.assert_allclose(
        model.cb(8.0, 1, on="sf", method="lr"),
        np.ravel(model.models[1].cb(8.0, on="sf", method="lr")),
    )


def test_cb_refusals(model):
    with pytest.raises(ValueError, match="pass `event`"):
        model.cb(TIMES)
    with pytest.raises(ValueError, match="Wald bounds only"):
        model.cb(TIMES, 1, method="lr")
    with pytest.raises(ValueError, match="Wald bounds only"):
        model.cb(TIMES, on="sf", method="bootstrap")
    with pytest.raises(ValueError, match="'on'"):
        model.cb(TIMES, 1, on="hf")
    with pytest.raises(ValueError, match="'method'"):
        model.cb(TIMES, 1, method="exact")
    with pytest.raises(ValueError, match="'bound'"):
        model.cb(TIMES, 1, bound="both")
    with pytest.raises(ValueError, match="cause"):
        model.cb(TIMES, 3)


def _draws_vs_delta(model, seed):
    """The SD of logit cif_1(8) and log H(8) over 2000 parameter draws
    from the covariance, each through with_params, and the delta-method
    standard errors on the same scales (read from the 95% bounds)."""
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(
        model._estimates(), model.covariance(), size=2000
    )
    t = 8.0
    values = []
    for d in draws:
        try:
            other = model.with_params(d)
        except ValueError:
            continue
        values.append((logit(other.cif(t, 1)), np.log(other.Hf(t))))
    spread = np.std(np.array(values), axis=0, ddof=1)
    zq = norm.ppf(0.975)
    cif_cb = model.cb(t, 1)
    H_cb = model.cb(t, on="Hf")
    delta = np.array(
        [
            (logit(cif_cb[1]) - logit(cif_cb[0])) / (2 * zq),
            (np.log(H_cb[1]) - np.log(H_cb[0])) / (2 * zq),
        ]
    )
    return spread, delta, len(values)


@pytest.mark.xfail(
    strict=True,
    reason="Found as designed (seed, 2000 draws and rtol 0.1 fixed "
    "beforehand): with 40 units the second cause's scale has a 53% "
    "standard error, 53 draws are negative and refused, and draws near 0 "
    "send logit cif to -50; the draws' SDs are 1.26 and 0.40 against "
    "delta SEs 0.33 and 0.20 (their central 68% half-widths 0.37 and "
    "0.23). The normal approximation, not the gradient, fails here; the "
    "n = 300 test below and the study agree to 2-3%.",
)
def test_with_params_draws_agree_with_the_delta_method(model):
    # (seed and tolerance fixed before the result was seen)
    spread, delta, kept = _draws_vs_delta(model, 20261105)
    assert kept > 1900
    np.testing.assert_allclose(spread, delta, rtol=0.1)


def test_with_params_draws_agree_with_the_delta_method_at_n_300():
    # The first data set of the coverage study's weibull2 scenario at
    # n = 300 (scripts/studies/parametric_competing_risks_cb_coverage.py,
    # its seed), where the study found the draws' SD 1.02 to 1.03 times
    # the delta SE; added after the n = 40 test above was found to fail.
    rng = np.random.default_rng(20261105 + 10**6)
    n = 300
    latent = np.column_stack(
        [10.0 * rng.weibull(2.0, size=n), 14.0 * rng.weibull(1.0, size=n)]
    )
    first = latent.min(axis=1)
    cause = np.where(latent[:, 0] <= latent[:, 1], 1, 2)
    cens = rng.uniform(0, 30.0, size=n)
    x = np.minimum(first, cens)
    e = np.where(first <= cens, cause, None).astype(object)
    m = ParametricCompetingRisks.fit(x, e, dist=Weibull)
    spread, delta, kept = _draws_vs_delta(m, 20261105)
    assert kept == 2000
    np.testing.assert_allclose(spread, delta, rtol=0.1)


# ---------------------------------------------------------------------------
# R flexsurv
# ---------------------------------------------------------------------------
R_FITS = {
    # cause: (loglik, alpha, beta, var alpha, cov, var beta)
    1: (
        -52.7849712461,
        11.5485410700,
        1.9286677969,
        2.2444389761e00,
        -1.9814983368e-02,
        1.1045095819e-01,
    ),
    2: (
        -48.2381981865,
        26.0245364329,
        0.6699929173,
        1.9345601537e02,
        -1.4409105641e00,
        2.6825939632e-02,
    ),
}
R_CIF = {
    # t: (cif1, cif2, sf), pmatrix.fs
    4.0: (0.0985834770, 0.2407772006, 0.6606393224),
    8.0: (0.2824658666, 0.3293608242, 0.3881733092),
    15.0: (0.5216612664, 0.3827012237, 0.0956375099),
}
R_CI = {
    # t: ((cif1 lower, upper), (cif2 lower, upper), (sf lower, upper))
    4.0: ((0.0368, 0.1916), (0.1357, 0.3852), (0.5087, 0.7729)),
    8.0: ((0.1533, 0.4186), (0.2053, 0.5294), (0.2270, 0.5217)),
    15.0: ((0.3224, 0.6731), (0.2457, 0.6097), (0.0184, 0.1966)),
}


def test_per_cause_fits_match_flexsurv(model):
    for k, (loglik, alpha, beta, v11, v12, v22) in R_FITS.items():
        m = model.models[k]
        np.testing.assert_allclose(m.params, [alpha, beta], rtol=1e-6)
        np.testing.assert_allclose(-m.neg_ll(), loglik, rtol=1e-8)
        np.testing.assert_allclose(
            m.covariance(), [[v11, v12], [v12, v22]], rtol=1e-4
        )
    names = model.covariance_names
    assert names.index("2: alpha") == 2


def test_cif_and_sf_match_flexsurv_multistate(model):
    for t, (cif1, cif2, sf) in R_CIF.items():
        np.testing.assert_allclose(model.cif(t, 1), cif1, rtol=1e-6)
        np.testing.assert_allclose(model.cif(t, 2), cif2, rtol=1e-6)
        np.testing.assert_allclose(model.sf(t), sf, rtol=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="Found as designed (atol 0.03 fixed beforehand): 15 of the 18 "
    "ends agree within 0.03; three do not, the cause-2 upper ends at t = 8 "
    "and 15 (0.487 and 0.556 against flexsurv's 0.529 and 0.610) and the "
    "cause-1 lower end at 15 (0.357 against 0.322). Cause 2's scale has a "
    "53% standard error with 40 units, where flexsurv's percentiles of "
    "draws on the log scale are skewed and the delta method on the logit "
    "is not: two methods, not one computed two ways. The coverage study "
    "checks the Wald bounds themselves.",
)
def test_bounds_near_flexsurv_simulated_intervals(model):
    # flexsurv's intervals are percentiles of 20000 simulated parameter
    # vectors (normal on the log scale), not the delta method: loose.
    for t, (ci1, ci2, ci_sf) in R_CI.items():
        np.testing.assert_allclose(model.cb(t, 1), ci1, atol=0.03)
        np.testing.assert_allclose(model.cb(t, 2), ci2, atol=0.03)
        np.testing.assert_allclose(model.cb(t, on="sf"), ci_sf, atol=0.03)
