"""The test of two survival curves at one time, ``sf_diff``."""

import numpy as np
import pytest
from scipy.stats import chi2

import surpyval as sp

# The AML maintenance data (R's survival::aml).
X = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161]
X += [5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
C = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1] + [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


@pytest.fixture
def aml():
    a = sp.KaplanMeier.fit(X[:11], c=C[:11])
    b = sp.KaplanMeier.fit(X[11:], c=C[11:])
    return a, b


def test_log_log_matches_lifelines_at_an_observed_time(aml):
    # lifelines 0.30.3, survival_difference_at_fixed_point_in_time_test(23,
    # kmf_a, kmf_b): 0.3474236632287309, p = 0.5555751674863638. 23 is an
    # observed time of both groups; between observed times lifelines
    # interpolates the Greenwood variance linearly while the estimate is a
    # step, so its numbers there differ from these.
    res = sp.sf_diff(*aml, 23)
    np.testing.assert_allclose(
        res["statistic"], 0.3474236632287309, rtol=1e-10
    )
    np.testing.assert_allclose(res["p_value"], 0.5555751674863638, rtol=1e-10)


def _greenwood(x, c, t):
    """The Kaplan-Meier estimate at t and Greenwood's sum, by hand."""
    x, c = np.asarray(x, float), np.asarray(c)
    s, g = 1.0, 0.0
    for u in np.unique(x[(c == 0) & (x <= t)]):
        r = np.sum(x >= u)
        d = np.sum((x == u) & (c == 0))
        s *= 1 - d / r
        g += d / (r * (r - d))
    return s, g


@pytest.mark.parametrize(
    "transform", ["log-log", "log", "linear", "arcsine-sqrt", "logit"]
)
@pytest.mark.parametrize("t", [12.0, 23.0, 30.0])
def test_every_transform_against_klein_et_al_formulae(aml, transform, t):
    # Klein et al. (2007), section 2: the statistic on each scale, with
    # the delta-method variances written out from Greenwood's sum.
    s1, g1 = _greenwood(X[:11], C[:11], t)
    s2, g2 = _greenwood(X[11:], C[11:], t)
    if transform == "linear":
        num, den = s1 - s2, g1 * s1**2 + g2 * s2**2
    elif transform == "log":
        num, den = np.log(s1) - np.log(s2), g1 + g2
    elif transform == "log-log":
        num = np.log(-np.log(s1)) - np.log(-np.log(s2))
        den = g1 / np.log(s1) ** 2 + g2 / np.log(s2) ** 2
    elif transform == "arcsine-sqrt":
        num = np.arcsin(np.sqrt(s1)) - np.arcsin(np.sqrt(s2))
        den = g1 * s1 / (4 * (1 - s1)) + g2 * s2 / (4 * (1 - s2))
    else:
        num = np.log(s1 / (1 - s1)) - np.log(s2 / (1 - s2))
        den = g1 / (1 - s1) ** 2 + g2 / (1 - s2) ** 2
    res = sp.sf_diff(*aml, t, transform=transform)
    np.testing.assert_allclose(res["statistic"], num**2 / den, rtol=1e-12)
    np.testing.assert_allclose(
        res["p_value"], chi2.sf(num**2 / den, 1), rtol=1e-10
    )
    np.testing.assert_allclose(res["difference"], s1 - s2, rtol=1e-12)
    np.testing.assert_allclose(
        res["se"], np.sqrt(g1 * s1**2 + g2 * s2**2), rtol=1e-12
    )
    assert res["lower"] < res["difference"] < res["upper"]


def test_between_observed_times_it_holds_the_last_step(aml):
    assert sp.sf_diff(*aml, 29.5) == {**sp.sf_diff(*aml, 28), "x": 29.5}


def test_swapping_the_groups_flips_only_the_sign(aml):
    a, b = aml
    ab, ba = sp.sf_diff(a, b, 30), sp.sf_diff(b, a, 30)
    assert ab["statistic"] == pytest.approx(ba["statistic"], rel=1e-14)
    assert ab["difference"] == -ba["difference"]


def test_nelson_aalen_uses_its_own_variance():
    a = sp.NelsonAalen.fit(X[:11], c=C[:11])
    b = sp.NelsonAalen.fit(X[11:], c=C[11:])
    res = sp.sf_diff(a, b, 23)
    i_a = np.searchsorted(a.x, 23, side="right") - 1
    i_b = np.searchsorted(b.x, 23, side="right") - 1
    num = np.log(-np.log(a.R[i_a])) - np.log(-np.log(b.R[i_b]))
    den = (
        a.greenwood[i_a] / np.log(a.R[i_a]) ** 2
        + b.greenwood[i_b] / np.log(b.R[i_b]) ** 2
    )
    np.testing.assert_allclose(res["statistic"], num**2 / den, rtol=1e-12)


def test_refusals_say_what_to_do(aml):
    a, b = aml
    with pytest.raises(ValueError, match="transform"):
        sp.sf_diff(a, b, 23, transform="probit")
    # Before any failure there is nothing to compare.
    with pytest.raises(ValueError, match="later time"):
        sp.sf_diff(a, b, 1, transform="linear")
    # log(-log(1)) is not defined.
    with pytest.raises(ValueError, match="transform='linear'"):
        sp.sf_diff(a, b, 1)
    # Past b's last time it is 0 there, where log-log is not defined
    with pytest.raises(ValueError, match="model_b"):
        sp.sf_diff(a, b, 50)
    # Past a's last time with its estimate above 0
    short = sp.KaplanMeier.fit([1, 2, 3, 4], c=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="past model_a's largest"):
        sp.sf_diff(short, b, 10)
    with pytest.raises(ValueError, match="'x'"):
        sp.sf_diff(a, b, "soon")
    with pytest.raises(ValueError, match="'x'"):
        sp.sf_diff(a, b, np.inf)
    ecdf = sp.NonParametric.fit_from_ecdf([1, 2, 3], [0.9, 0.5, 0.2])
    with pytest.raises(ValueError, match="no variance"):
        sp.sf_diff(ecdf, b, 2)


# -- gaps found by the 0.26 test audit ---------------------------------------


def _weighted_aml(scale):
    """The two AML groups fitted with fixed sampling weights times
    ``scale``."""
    x, c = np.asarray(X, float), np.asarray(C)
    w_a = scale * (1 + (np.arange(11) % 3) / 2)
    w_b = scale * (1 + (np.arange(12) % 3) / 2)
    a = sp.KaplanMeier.fit(x[:11], c=c[:11], w=w_a)
    b = sp.KaplanMeier.fit(x[11:], c=c[11:], w=w_b)
    return a, b


@pytest.mark.filterwarnings("ignore:The weights leave an effective sample")
def test_weighted_fits_use_the_robust_variance_whatever_the_weight_scale():
    """Audit C2. With weights, sf_diff uses each weighted fit's robust
    variance of log S (its ``greenwood``), which, like the estimate, does
    not change when every weight is multiplied by 10."""
    a, b = _weighted_aml(1.0)
    res = sp.sf_diff(a, b, 23)
    res10 = sp.sf_diff(*_weighted_aml(10.0), 23)
    for key in ("statistic", "p_value", "difference", "se", "lower", "upper"):
        np.testing.assert_allclose(res10[key], res[key], rtol=1e-10)
    i_a = np.searchsorted(a.x, 23, side="right") - 1
    i_b = np.searchsorted(b.x, 23, side="right") - 1
    s_a, s_b = a.R[i_a], b.R[i_b]
    v_a, v_b = a.greenwood[i_a], b.greenwood[i_b]
    np.testing.assert_allclose(
        res["se"], np.sqrt(v_a * s_a**2 + v_b * s_b**2), rtol=1e-12
    )
    np.testing.assert_allclose(res["difference"], s_a - s_b, rtol=1e-12)
    # The weights change the test: it is not the unweighted one.
    plain = sp.sf_diff(
        sp.KaplanMeier.fit(X[:11], c=C[:11]),
        sp.KaplanMeier.fit(X[11:], c=C[11:]),
        23,
    )
    assert not np.isclose(res["statistic"], plain["statistic"], rtol=1e-6)
