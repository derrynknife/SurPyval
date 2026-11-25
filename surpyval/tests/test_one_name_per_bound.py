"""One name per interval scale and per bound (v0.30).

The scales of the confidence bounds had two names each: ``"exp"`` on the
Kaplan-Meier, multi-state and competing-risks models and ``"log-log"`` on
the Cox curve for the same log(-log) scale, ``"normal"`` and ``"plain"``
for the estimate itself; ``sf_diff`` had ``"linear"`` and
``"arcsine-sqrt"``. The names are now R's ``conf.type`` names. A few bound
methods had names of their own (``R_cb``, ``parameter_cb``,
``bootstrap_ci``), and seven functions took the level as ``alpha``. Every
old spelling still works until v0.31, with a ``FutureWarning`` naming the
new one, and gives the same numbers to the last digit.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import surpyval as sp

X = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0])
C = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
Q = [2.5, 4.0, 6.5]


def _quiet(call):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        return call()


def _old(call, match):
    with pytest.warns(FutureWarning, match=match) as caught:
        out = call()
    # It points at the caller's line, not inside the package
    assert caught[0].filename == __file__
    return out


@pytest.fixture(scope="module")
def km():
    return sp.KaplanMeier.fit(X, c=C)


@pytest.mark.parametrize("old, new", [("exp", "log-log"), ("normal", "plain")])
def test_kaplan_meier_cb_scale(km, old, new):
    got = _old(
        lambda: km.cb(Q, bound_type=old),
        f"KaplanMeier.cb: bound_type='{old}' is deprecated .* v0.31; use "
        f"bound_type='{new}'",
    )
    assert_array_equal(got, _quiet(lambda: km.cb(Q, bound_type=new)))


def test_the_default_is_log_log(km):
    assert_array_equal(
        _quiet(lambda: km.cb(Q)),
        _quiet(lambda: km.cb(Q, bound_type="log-log")),
    )


@pytest.mark.parametrize("old, new", [("exp", "log-log"), ("normal", "plain")])
def test_band_and_quantile_cb_scale(km, old, new):
    got = _old(lambda: km.band(bound_type=old), "KaplanMeier.band")
    assert_array_equal(got, _quiet(lambda: km.band(bound_type=new)))
    got = _old(lambda: km.quantile_cb(0.3, bound_type=old), "quantile_cb")
    assert_array_equal(
        got, _quiet(lambda: km.quantile_cb(0.3, bound_type=new))
    )


def test_r_cb_is_cb_reversed_with_a_warning(km):
    got = _old(lambda: km.R_cb(Q), r"R_cb is deprecated .* cb\(x, on='sf'\)")
    assert_array_equal(got, np.fliplr(_quiet(lambda: km.cb(Q, on="sf"))))
    got = _old(lambda: km.R_cb(Q, bound="upper"), "R_cb")
    assert_array_equal(got, _quiet(lambda: km.cb(Q, bound="upper")))


def test_plot_data_does_not_warn(km):
    data = _quiet(lambda: km.get_plot_data())
    assert_array_equal(data["cbs"], np.fliplr(km.cb(km.x)))


@pytest.mark.parametrize(
    "old, new", [("linear", "plain"), ("arcsine-sqrt", "arcsine")]
)
def test_sf_diff_transform(old, new):
    a = sp.KaplanMeier.fit(X, c=C)
    b = sp.KaplanMeier.fit(X * 1.3, c=C)
    got = _old(
        lambda: sp.sf_diff(a, b, 5.0, transform=old),
        f"sf_diff: transform='{old}' is deprecated",
    )
    want = _quiet(lambda: sp.sf_diff(a, b, 5.0, transform=new))
    assert got["statistic"] == want["statistic"]
    assert got["p_value"] == want["p_value"]
    assert got["transform"] == new


def test_competing_risks_cif_scale():
    e = np.array(
        ["a", "b", None, "a", "b", None, "a", "a", None, "b", "a", None]
    )
    model = sp.CompetingRisks.fit(X, e)
    for old, new in (("exp", "log-log"), ("normal", "plain")):
        got = _old(
            lambda: model.cb(Q, "a", bound_type=old), "CompetingRisks.cb"
        )
        assert_array_equal(
            got, _quiet(lambda: model.cb(Q, "a", bound_type=new))
        )


def test_mcf_exp_was_the_log_scale():
    # The MCF's 'exp' was the log scale, not the log(-log) one.
    x = [1, 3, 5, 2, 6, 4, 7, 9]
    i = [1, 1, 1, 2, 2, 3, 3, 3]
    model = sp.NonParametricCounting.fit(x, i=i)
    got = _old(
        lambda: model.mcf_cb(4.0, bound_type="exp"), "use bound_type='log'"
    )
    assert_array_equal(
        got, _quiet(lambda: model.mcf_cb(4.0, bound_type="log"))
    )
    got = _old(lambda: model.mcf_cb(4.0, bound_type="normal"), "'plain'")
    assert_array_equal(
        got, _quiet(lambda: model.mcf_cb(4.0, bound_type="plain"))
    )
    assert_array_equal(
        _quiet(lambda: model.mcf_cb(4.0)),
        _quiet(lambda: model.mcf_cb(4.0, bound_type="log")),
    )


def test_cox_curve_scale():
    Z = np.array([[0.0], [1.0]] * 6)
    model = _quiet(lambda: sp.CoxPH.fit(X, Z, c=C))
    got = _old(
        lambda: model.cb(Q, [1.0], bound_type="normal"),
        "bound_type='normal' is deprecated",
    )
    assert_array_equal(got, model.cb(Q, [1.0], bound_type="plain"))


def test_an_unknown_scale_is_still_refused(km):
    with pytest.raises(ValueError, match="bound_type"):
        km.cb(Q, bound_type="probit")


@pytest.fixture(scope="module")
def buckley_james():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(60, 1))
    t = np.exp(2.0 - 0.5 * Z[:, 0] + rng.normal(0, 0.5, 60))
    c = (t > 12).astype(int)
    return sp.BuckleyJames.fit(np.minimum(t, 12), Z, c=c)


def test_buckley_james_param_cb(buckley_james):
    m = buckley_james
    old = _old(
        lambda: m.bootstrap_ci(0.1, n_boot=30, random_state=1),
        r"BuckleyJamesModel.bootstrap_ci is deprecated .* param_cb",
    )
    new = _quiet(lambda: m.param_cb(alpha_ci=0.1, n_boot=30, random_state=1))
    assert_array_equal(old, new)
    # one coefficient, and one side at alpha is the two-sided end at 2 alpha
    name = m.parameter_names[0]
    assert_array_equal(
        m.param_cb(name, alpha_ci=0.1, n_boot=30, random_state=1), new[0]
    )
    lower = m.param_cb(alpha_ci=0.05, bound="lower", n_boot=30, random_state=1)
    assert_array_equal(lower, new[:, 0])
    with pytest.raises(ValueError, match="Unknown coefficient"):
        m.param_cb("nope")


def test_quantile_regression_param_cb():
    rng = np.random.default_rng(0)
    z = rng.uniform(0, 2, 200)
    t = np.exp(1.0 + 0.5 * z + 0.3 * rng.normal(size=200))
    c = (t > 20).astype(int)
    m = sp.QuantileRegression.fit(np.minimum(t, 20), z, c=c)
    old = _old(
        lambda: m.bootstrap_ci(0.5, 0.1, n_boot=30, random_state=1),
        r"QuantileRegressionModel.bootstrap_ci is deprecated",
    )
    new = _quiet(
        lambda: m.param_cb(alpha_ci=0.1, n_boot=30, random_state=1, tau=0.5)
    )
    assert_array_equal(old, new)
    upper = m.param_cb(
        "Intercept", 0.05, "upper", n_boot=30, random_state=1, tau=0.5
    )
    assert upper == new[0, 1]


def test_distributional_param_cb_with_rows():
    rng = np.random.default_rng(3)
    load = rng.uniform(0, 1, 200)
    x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
    m = sp.WeibullDistributional.fit(x, load)
    rows = [[0.0], [1.0]]
    old = _old(
        lambda: m.parameter_cb("beta", rows),
        r"parameter_cb is deprecated .* param_cb\(name, Z=Z",
    )
    assert_array_equal(old, _quiet(lambda: m.param_cb("beta", Z=rows)))
    # without Z it is the fitted parameter's bound, as before
    name = m.parameter_names[0]
    assert np.shape(m.param_cb(name)) == (2,)


def test_diagnose_alpha():
    m = sp.CoxPH.fit(X, np.array([[0.0], [1.0]] * 6), c=C)
    with pytest.warns(FutureWarning, match="'alpha' is deprecated .* v0.31"):
        old = m.diagnose(alpha=0.01)
    new = _quiet(lambda: m.diagnose(alpha_ci=0.01))
    assert old.table.equals(new.table)


def test_concordance_alpha():
    from surpyval.metrics import compare_concordance

    a, b = -X, -np.sqrt(X) + 0.1 * C
    with pytest.warns(FutureWarning, match="use 'alpha_ci'"):
        old = compare_concordance(X, C, a, b, alpha=0.1)
    new = _quiet(lambda: compare_concordance(X, C, a, b, alpha_ci=0.1))
    assert str(old) == str(new)
    with pytest.raises(ValueError, match="pass 'alpha_ci' only"):
        compare_concordance(X, C, a, b, alpha=0.1, alpha_ci=0.1)
