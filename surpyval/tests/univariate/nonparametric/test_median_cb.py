"""The median's confidence interval of the non-parametric models
(``median_cb``) and the one-sided ``quantile_cb``, against R's
``quantile(survfit(...))`` (survival 3.x)."""

import numpy as np
import pytest

import surpyval as sp

# R: f <- survfit(Surv(time, status) ~ 1, data = lung, conf.type = ct,
#    conf.int = level); quantile(f, c(.25, .5, .75))
_R_LUNG = {
    ("exp", 0.05): ([144, 284, 457], [194, 361, 643]),
    ("exp", 0.10): ([145, 285, 473], [186, 353, 641]),
    ("log", 0.05): ([145, 285, 460], [197, 363, 654]),
    ("log", 0.10): ([153, 285, 473], [189, 353, 641]),
    ("normal", 0.05): ([145, 284, 457], [197, 361, 643]),
    ("normal", 0.10): ([147, 285, 460], [186, 353, 641]),
}

# R's aml data (survival::aml), status 1 a death
_AML_T = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161]
_AML_T += [5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
_AML_S = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0]
_AML_S += [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]


@pytest.fixture(scope="module")
def lung():
    df = sp.datasets.load_lung()
    return sp.KaplanMeier.fit(
        df["time"].to_numpy(float), c=1 - df["status"].to_numpy(int)
    )


def _aml():
    return sp.KaplanMeier.fit(_AML_T, c=1 - np.array(_AML_S))


@pytest.mark.parametrize("key", sorted(_R_LUNG))
def test_quantile_cb_matches_r_on_lung(lung, key):
    bound_type, alpha = key
    lower, upper = _R_LUNG[key]
    out = lung.quantile_cb([0.25, 0.5, 0.75], alpha, bound_type)
    np.testing.assert_array_equal(out[:, 0], lower)
    np.testing.assert_array_equal(out[:, 1], upper)
    np.testing.assert_array_equal(
        lung.median_cb(alpha, bound_type=bound_type), [lower[1], upper[1]]
    )


def test_median_cb_matches_r_on_aml():
    model = _aml()
    assert model.median == 27
    # R conf.type = "log-log" (the default here) and "plain"
    np.testing.assert_array_equal(model.median_cb(), [13, 34])
    np.testing.assert_array_equal(
        model.median_cb(bound_type="normal"), [18, 34]
    )
    np.testing.assert_array_equal(model.median_cb(alpha_ci=0.1), [18, 34])
    # R's conf.type = "log" gives 48 for the upper end: its upper survival
    # bound is not monotone there (0.458 at 45, 0.462 at 48) and R's
    # findq reads it as if sorted. Both are below 0.5, so the set of times
    # whose interval contains 0.5 ends at 45, the first of them.
    np.testing.assert_array_equal(model.median_cb(bound_type="log"), [18, 45])


def test_one_sided_bounds_are_the_ends_at_twice_alpha(lung):
    p = [0.25, 0.5, 0.75]
    two = lung.quantile_cb(p, 0.1)
    np.testing.assert_array_equal(
        lung.quantile_cb(p, 0.05, bound="lower"), two[:, 0]
    )
    np.testing.assert_array_equal(
        lung.quantile_cb(p, 0.05, bound="upper"), two[:, 1]
    )
    assert lung.median_cb(0.05, bound="lower") == two[1, 0]
    assert lung.median_cb(0.05, bound="upper") == two[1, 1]
    assert isinstance(lung.median_cb(bound="upper"), float)


def test_open_interval_and_refusals():
    model = sp.KaplanMeier.fit(
        [1, 2, 3, 4, 5, 6, 7, 8], c=[0, 1, 0, 0, 1, 0, 0, 1]
    )
    lower, upper = model.median_cb()
    assert lower == 1 and np.isnan(upper)
    with pytest.raises(ValueError, match="bound"):
        model.median_cb(bound="both")
    with pytest.raises(ValueError, match="below 0.5"):
        model.median_cb(alpha_ci=0.6, bound="lower")


@pytest.mark.parametrize(
    "cls", ["NelsonAalen", "FlemingHarrington", "Turnbull"]
)
def test_every_nonparametric_model_has_it(cls):
    model = getattr(sp, cls).fit(_AML_T, c=1 - np.array(_AML_S))
    np.testing.assert_array_equal(model.median_cb(), model.quantile_cb(0.5))
    assert model.median_cb()[0] <= model.median <= model.median_cb()[1]
