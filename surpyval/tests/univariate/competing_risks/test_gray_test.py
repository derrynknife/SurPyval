"""Gray's test for comparing cumulative incidence functions (#216).

Gray's test is validated by simulation: under the null (identical cause-``k``
cumulative incidence across groups) its p-values are ~Uniform -- including
under independent censoring, which exercises the IPCW subdistribution
weighting -- and it is powerful against a genuine CIF difference.
"""

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import gray_test


def _sim_cr(seed, n, lam1_by_group, lam2=0.5, cens_rate=0.0, groups=2):
    """Two-cause competing-risks sample; cause-1 rate varies by group."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, groups, n)
    lam1 = np.asarray(lam1_by_group)[g]
    t1 = rng.exponential(1 / lam1)
    t2 = rng.exponential(1 / lam2, n)
    x = np.minimum(t1, t2)
    e = np.where(t1 < t2, 1, 2).astype(object)
    if cens_rate > 0:
        cens = rng.exponential(1 / cens_rate, n)
        obs = x <= cens
        x = np.where(obs, x, cens)
        e = np.where(obs, e, None)
    return x, e, g


def _null_pvalues(cens_rate=0.0, groups=2, reps=120):
    out = []
    for s in range(reps):
        x, e, g = _sim_cr(
            s, 200, [1.0] * groups, cens_rate=cens_rate, groups=groups
        )
        out.append(sp.gray_test(x, e, g, event=1).p_value)
    return np.array(out)


def test_gray_null_calibration_no_censoring():
    p = _null_pvalues(cens_rate=0.0)
    assert (p < 0.05).mean() < 0.12
    assert 0.4 < p.mean() < 0.6


def test_gray_null_calibration_with_censoring():
    # The IPCW subdistribution weighting must keep the test calibrated when
    # observations are independently censored.
    p = _null_pvalues(cens_rate=0.4)
    assert (p < 0.05).mean() < 0.12
    assert 0.4 < p.mean() < 0.6


def test_gray_null_calibration_three_groups():
    p = _null_pvalues(cens_rate=0.0, groups=3)
    assert (p < 0.05).mean() < 0.12
    result = sp.gray_test(*_sim_cr(0, 200, [1.0, 1.0, 1.0], groups=3), event=1)
    assert result.df == 2


def test_gray_detects_cif_difference():
    x, e, g = _sim_cr(1, 400, [2.0, 0.5], cens_rate=0.0)
    result = sp.gray_test(x, e, g, event=1)
    assert result.p_value < 0.01
    assert result.df == 1
    assert result.cause == 1


def test_gray_censoring_inferred_from_none():
    # e is None for censored; passing c explicitly must give the same result.
    x, e, g = _sim_cr(2, 200, [1.5, 0.8], cens_rate=0.3)
    c = np.array([1 if ei is None else 0 for ei in e])
    r_inferred = sp.gray_test(x, e, g, event=1)
    r_explicit = sp.gray_test(x, e, g, event=1, c=c)
    assert r_inferred.p_value == pytest.approx(r_explicit.p_value)


def test_gray_requires_two_groups():
    x, e, g = _sim_cr(3, 100, [1.0], groups=1)
    with pytest.raises(ValueError, match="two groups"):
        sp.gray_test(x, e, g, event=1)


def test_gray_unknown_cause_raises():
    x, e, g = _sim_cr(4, 100, [1.0, 1.0])
    with pytest.raises(ValueError, match="No events of cause"):
        sp.gray_test(x, e, g, event=99)


# ---------------------------------------------------------------------------
# A NaN cause is read as censored when ``c`` is omitted, the
# missing-cause rule of every other competing-risks class.
# ---------------------------------------------------------------------------


X8 = [1, 2, 3, 4, 5, 6, 7, 8]
G8 = [0, 0, 0, 0, 1, 1, 1, 1]


def test_gray_test_nan_cause_is_censored():
    with_none = gray_test(X8, [1, 2, None, 1, 2, 1, None, 2], G8, event=1)
    with_nan = gray_test(X8, [1, 2, np.nan, 1, 2, 1, np.nan, 2], G8, event=1)
    assert with_nan.statistic == pytest.approx(with_none.statistic)
    assert with_nan.p_value == pytest.approx(with_none.p_value)
    # and both match the explicit censoring flags
    c = [0, 0, 1, 0, 0, 0, 1, 0]
    explicit = gray_test(X8, [1, 2, None, 1, 2, 1, None, 2], G8, 1, c=c)
    assert with_nan.statistic == pytest.approx(explicit.statistic)


def test_gray_test_pandas_missing_cause_is_censored():
    frame = pd.DataFrame(
        {"x": X8, "e": [1, 2, None, 1, 2, 1, None, 2], "g": G8}
    )
    assert frame["e"].isna().sum() == 2  # stored as NaN by pandas
    res = gray_test(frame["x"], frame["e"], frame["g"], event=1)
    ref = gray_test(X8, [1, 2, None, 1, 2, 1, None, 2], G8, event=1)
    assert res.statistic == pytest.approx(ref.statistic)


def test_gray_test_c_must_agree_with_missing_causes():
    # a NaN cause on a row flagged as a failure is as ambiguous here as it is
    # for the model classes
    with pytest.raises(ValueError, match="missing event"):
        gray_test(
            X8,
            [1, 2, np.nan, 1, 2, 1, np.nan, 2],
            G8,
            event=1,
            c=np.zeros(8),
        )


# ---------------------------------------------------------------------------
# Gray's (1988) construction: each group's subdistribution risk
# set (and so its censoring distribution) is estimated from the
# group's own data, with Gray's asymptotic variance, so the test
# stays calibrated when the groups are censored differently.
# Argument checks.
# ---------------------------------------------------------------------------


def _sim_cr_gray(rng, n, h1, h2, cens_mean):
    t1 = rng.exponential(1 / h1, n)
    t2 = rng.exponential(1 / h2, n)
    cz = rng.exponential(cens_mean, n)
    x = np.minimum.reduce([t1, t2, cz])
    e = np.where(x == cz, None, np.where(t1 < t2, 1, 2)).astype(object)
    return x, e


@pytest.mark.parametrize("cause", [1, 2])
def test_gray_calibrated_under_unequal_censoring(cause):
    # Identical cause-specific hazards, censoring means 2 and 50. With one
    # pooled censoring estimate about half of these tests rejected at 5%.
    rng = np.random.default_rng(4)
    stats = []
    for _ in range(150):
        x0, e0 = _sim_cr_gray(rng, 400, 0.1, 0.2, 2.0)
        x1, e1 = _sim_cr_gray(rng, 400, 0.1, 0.2, 50.0)
        res = gray_test(
            np.concatenate([x0, x1]),
            np.concatenate([e0, e1]),
            np.repeat([0, 1], 400),
            event=cause,
        )
        stats.append(res.statistic)
    stats = np.array(stats)
    assert np.mean(stats > 3.841) < 0.1
    # a chi-square(1) statistic has mean 1
    assert 0.75 < stats.mean() < 1.3


def test_gray_equal_censoring_close_to_pooled_version():
    # The docstring example: the pooled-censoring version gave 19.878, and
    # SurPyval's own variance 19.962; cmprsk::cuminc gives 20.112602058
    # (#380).
    rng = np.random.default_rng(0)
    group = rng.binomial(1, 0.5, 200)
    t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * group)))
    t_b = rng.exponential(1 / 0.05, 200)
    t_c = rng.uniform(0, 20, 200)
    x = np.minimum(np.minimum(t_a, t_b), t_c).round(3)
    first = np.where(t_a < t_b, "a", "b")
    e = np.where(t_c < np.minimum(t_a, t_b), None, first)
    res = gray_test(x, e, group, event="a")
    assert res.statistic == pytest.approx(20.112602058, rel=1e-9)


def test_gray_invariances_hold():
    rng = np.random.default_rng(3)
    x, e = _sim_cr_gray(rng, 240, 0.1, 0.2, 10.0)
    g = np.repeat([0, 1, 2], 80)
    n = rng.integers(1, 4, 240)
    base = gray_test(x, e, g, event=1, n=n)
    expanded = gray_test(
        np.repeat(x, n), np.repeat(e, n), np.repeat(g, n), event=1
    )
    relabelled = gray_test(x, e, np.array(["b", "c", "a"])[g], event=1, n=n)
    assert expanded.statistic == pytest.approx(base.statistic)
    assert relabelled.statistic == pytest.approx(base.statistic)


X6 = [1, 2, 3, 4, 5, 6]
E6 = [1, 2, 1, 2, 1, 2]
G6 = [0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"c": [-1, 0, 0, 0, 0, 0]}, "c must be 0"),
        ({"c": [2, 0, 0, 0, 0, 0]}, "c must be 0"),
        ({"n": [0, 1, 1, 1, 1, 1]}, "n must be finite and positive"),
        ({"n": [-1, 1, 1, 1, 1, 1]}, "n must be finite and positive"),
        ({"n": [1, 1]}, "one count per time"),
        ({"rho": np.nan}, "rho"),
    ],
)
def test_gray_rejects_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        gray_test(X6, E6, G6, event=1, **kwargs)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_gray_rejects_non_finite_times(bad):
    with pytest.raises(ValueError, match="finite"):
        gray_test([bad, 2, 3, 4, 5, 6], E6, G6, event=1)


@pytest.mark.parametrize("missing", [None, np.nan])
def test_gray_rejects_missing_group_labels(missing):
    with pytest.raises(ValueError, match="missing label"):
        gray_test(X6, E6, [missing, 0, 0, 1, 1, 1], event=1)


def test_gray_rejects_length_mismatches():
    with pytest.raises(ValueError, match="one label per time"):
        gray_test(X6, E6, G6[:-1], event=1)
    with pytest.raises(ValueError, match="one cause per time"):
        gray_test(X6, E6[:-1], G6, event=1)
    with pytest.raises(ValueError, match="empty"):
        gray_test([], [], [], event=1)


def test_gray_mixed_and_tuple_labels():
    res = gray_test(X6, E6, [0, "a", 0, "a", 0, "a"], event=1)
    assert res.groups == [0, "a"]
    causes = [("a", 1), ("b", 2)] * 3
    tup = gray_test(X6, causes, G6, event=("a", 1))
    assert tup.statistic == pytest.approx(gray_test(X6, E6, G6, 1).statistic)


# ---------------------------------------------------------------------------
# #380: Gray's test is cmprsk's.
# ---------------------------------------------------------------------------


_GRAY = dict(
    x=[2, 7, 6, 3, 1, 6, 3, 7, 3, 6, 2, 5, 1, 7, 1, 6, 7, 4, 7, 5, 7, 6, 8]
    + [1, 1, 6, 6, 6, 7, 4, 7, 5, 2, 2, 7, 6, 7, 8, 7, 3, 5, 4, 3, 8, 8],
    e=[1, 1, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 1, 2, 0, 2, 2, 0, 0, 0, 2, 1, 0]
    + [1, 1, 1, 2, 2, 1, 0, 0, 0, 2, 2, 0, 2, 1, 1, 1, 2, 2, 0, 2, 0, 2],
    group=[1, 1, 2, 0, 0, 1, 0, 2, 2, 1, 0, 2, 0, 2, 2, 0, 2, 2, 1, 2, 1, 1]
    + [1, 0, 1, 1, 0, 1, 0, 2, 0, 0, 1, 0, 1, 0, 2, 0, 2, 2, 2, 2, 1, 0, 1],
)


@pytest.mark.parametrize(
    "rho, event, expected",
    [
        (0, 1, 0.680791319807),
        (0, 2, 1.495728261106),
        (1, 1, 0.877130594184),
        (1, 2, 1.212189254797),
    ],
)
def test_gray_test_matches_cmprsk_on_three_tied_groups(rho, event, expected):
    # R cmprsk 2.2-11: cuminc(x, e, group, rho = rho)$Tests. The variance
    # was SurPyval's own linearisation (6.162 against cmprsk's 5.065 on
    # the reference suite's tied fixture), and the rho weight used a
    # different pooled incidence.
    e = [None if k == 0 else k for k in _GRAY["e"]]
    res = sp.gray_test(_GRAY["x"], e, _GRAY["group"], event=event, rho=rho)
    assert res.df == 2
    assert res.statistic == pytest.approx(expected, rel=1e-10)


def test_gray_test_counts_equal_repeated_rows():
    rng = np.random.default_rng(3)
    x = rng.integers(1, 6, 30).astype(float)
    e = rng.choice(np.array([None, "a", "b"], dtype=object), 30)
    g = rng.integers(0, 2, 30)
    n = rng.integers(1, 4, 30)
    counted = sp.gray_test(x, e, g, event="a", n=n, rho=0.5)
    rows = np.repeat(np.arange(30), n)
    repeated = sp.gray_test(x[rows], e[rows], g[rows], event="a", rho=0.5)
    assert counted.statistic == pytest.approx(repeated.statistic, rel=1e-12)
