"""Harrell's C in O(n log n) and ``model.concordance`` (#512).

``pairwise`` is the O(n^2) definition ``surpyval.utils.score.score`` used
until 0.22 (with the #276 tie conventions), kept here as the oracle: the
fast count must agree with it on every kind of tie.
"""

import time
import warnings
from itertools import combinations
from math import isclose

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.datasets import load_lung
from surpyval.metrics import concordance_index


def pairwise(x, c, scores, tie_tol=1e-8):
    concordance = 0.0
    permissible = 0
    rows = list(zip(np.asarray(x), np.asarray(c), np.asarray(scores)))
    for a, b in combinations(rows, 2):
        if (a[0], a[1]) > (b[0], b[1]):
            a, b = b, a
        x_1, c_1, s_1 = a
        x_2, c_2, s_2 = b
        if c_1 == 1 and x_1 != x_2:
            continue
        if x_1 == x_2 and c_1 == c_2 == 1:
            continue
        permissible += 1
        tied = isclose(s_1, s_2, abs_tol=tie_tol)
        if x_1 != x_2:
            if s_1 > s_2:
                concordance += 1
            elif tied:
                concordance += 0.5
        elif c_1 == 0 and c_2 == 0:
            concordance += 1 if tied else 0.5
        else:
            death, other = (s_1, s_2) if c_1 == 0 else (s_2, s_1)
            if tied:
                concordance += 0.5
            elif death > other:
                concordance += 1
    return concordance / permissible


def _sample(rng, n, kind):
    """Data with ties in time, in score, censoring, or all of them."""
    if kind in ("time_ties", "all_ties"):
        x = rng.integers(0, max(2, n // 4), n).astype(float)
    else:
        x = rng.exponential(10.0, n)
    c = (rng.random(n) < rng.uniform(0.0, 0.7)).astype(int)
    if kind in ("score_ties", "all_ties"):
        s = rng.integers(0, 5, n).astype(float)
    elif kind == "near_ties":
        # scores within the tolerance of each other, either side
        s = np.round(rng.normal(size=n), 1)
        s = s + rng.choice([0.0, 4e-9, -4e-9, 3e-8], n)
    elif kind == "large_scores":
        # the relative part of the tolerance (math.isclose's 1e-9)
        s = 1e3 * rng.integers(0, 4, n) + rng.choice([0.0, 5e-7, 5e-5], n)
    else:
        s = rng.normal(size=n)
    return x, c, s


KINDS: tuple[str, ...] = ("plain", "time_ties", "score_ties", "all_ties")
KINDS += ("near_ties", "large_scores")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n", [2, 3, 5, 17, 60, 250])
def test_agrees_with_pairwise_definition(n, kind):
    rng = np.random.default_rng(n * 31 + len(kind))
    for _ in range(8 if n < 100 else 2):
        x, c, s = _sample(rng, n, kind)
        try:
            expected = pairwise(x, c, s)
        except ZeroDivisionError:
            with pytest.raises(ValueError, match="No usable pairs"):
                concordance_index(x, c, s)
            continue
        assert concordance_index(x, c, s) == pytest.approx(
            expected, rel=1e-12, abs=1e-12
        )


@pytest.mark.parametrize("kind", ["all_ties", "near_ties"])
def test_agrees_with_pairwise_definition_n_2000(kind):
    rng = np.random.default_rng(2000)
    x, c, s = _sample(rng, 2000, kind)
    assert concordance_index(x, c, s) == pytest.approx(
        pairwise(x, c, s), rel=1e-12
    )


def test_fifty_thousand_subjects_in_well_under_a_second():
    # The pairwise count took about 3 s at 5,000 subjects and would take
    # about 5 minutes here; the fast one takes about 0.15 s.
    rng = np.random.default_rng(0)
    x, c, s = _sample(rng, 50_000, "plain")
    start = time.perf_counter()
    concordance_index(x, c, s)
    assert time.perf_counter() - start < 5.0


def test_lifelines_agrees_without_tied_times():
    # lifelines 0.30.3, concordance_index(x, -risk, 1 - c), on the lung
    # Cox model (age, sex, ph.ecog) with the time ties broken; it counts
    # tied event times differently (0.637135 against 0.636942 with them).
    # ``status`` is 1 for a death (#509): the censoring flag is 1 - status.
    lung = load_lung().dropna(subset=["ph.ecog"])
    lung["censored"] = 1 - lung["status"]
    cols = ["age", "sex", "ph.ecog"]
    model = sp.CoxPH.fit_from_df(
        lung, x_col="time", c_col="censored", Z_cols=cols
    )
    x = lung["time"].to_numpy() + np.arange(len(lung)) * 1e-6
    risk = lung[cols].to_numpy() @ model.beta
    c = lung["censored"].to_numpy()
    assert concordance_index(x, c, risk) == pytest.approx(
        0.6368729181, abs=1e-10
    )
    assert model.concordance() == pytest.approx(0.636942, abs=1e-6)


def test_missing_values_and_errors():
    assert np.isnan(concordance_index([1.0, np.nan, 3], [0, 0, 0], [3, 2, 1]))
    assert np.isnan(concordance_index([1.0, 2, 3], [0, 0, 0], [3, np.nan, 1]))
    with pytest.raises(ValueError, match="'c' must be 0"):
        concordance_index([1.0, 2, 3], [0, -1, 0], [3, 2, 1])
    with pytest.raises(ValueError, match="same length"):
        concordance_index([1.0, 2, 3], [0, 0, 0], [3, 2])
    with pytest.raises(ValueError, match="No usable pairs"):
        concordance_index([1.0, 2], [1, 1], [3, 2])


def test_old_name_warns_and_agrees():
    from surpyval.utils.score import score

    with pytest.warns(DeprecationWarning, match="concordance_index"):
        old = score([1.0, 2, 3, 4], [0, 1, 0, 0], [4.0, 1, 2, 3])
    assert old == concordance_index([1.0, 2, 3, 4], [0, 1, 0, 0], [4, 1, 2, 3])


def test_exported_from_the_package():
    assert sp.concordance_index is concordance_index
    assert sp.metrics.concordance_index is concordance_index


# ---------------------------------------------------------------------------
# model.concordance
# ---------------------------------------------------------------------------
def _data(n=120, seed=3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    t = sp.Weibull.random(n, 10.0, 1.5, random_state=rng) * np.exp(
        -(0.8 * z[:, 0] - 0.4 * z[:, 1]) / 1.5
    )
    cens = rng.uniform(0, 25, n)
    x = np.minimum(t, cens).round(3)
    c = (cens < t).astype(int)
    return x, c, z


@pytest.mark.parametrize(
    "fitter, sign",
    [
        (sp.CoxPH, 1.0),
        (sp.WeibullPH, 1.0),
        (sp.WeibullAFT, 1.0),
        (sp.WeibullPO, -1.0),
        (sp.AdditiveHazards, 1.0),
        (sp.BuckleyJames, -1.0),
    ],
    ids=["CoxPH", "WeibullPH", "WeibullAFT", "WeibullPO", "Lin-Ying", "BJ"],
)
def test_model_concordance_is_that_of_its_linear_predictor(fitter, sign):
    # The documented risk score of each family ranks as +-beta'Z.
    x, c, z = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fitter.fit(x, z, c)
    if hasattr(model, "beta"):
        beta = np.asarray(model.beta)
    else:
        beta = np.asarray(model.params)[model.k_dist :]
    expected = concordance_index(x, c, sign * z @ beta)
    assert model.concordance() == pytest.approx(expected, abs=1e-12)
    # and on new data, given explicitly
    x2, c2, z2 = _data(60, seed=9)
    expected = concordance_index(x2, c2, sign * z2 @ beta)
    assert model.concordance(x2, c2, z2) == pytest.approx(expected)


def test_model_concordance_counts_repeat_rows():
    x, c, z = _data(40)
    n = np.tile([1, 2], 20)
    model = sp.CoxPH.fit(x, z, c, n)
    rows = np.repeat(np.arange(40), n)
    expected = concordance_index(x[rows], c[rows], z[rows] @ model.beta)
    assert model.concordance() == pytest.approx(expected)


def test_model_concordance_takes_a_data_frame():
    x, c, z = _data()
    df = pd.DataFrame({"x": x, "c": c, "a": z[:, 0], "b": z[:, 1]})
    model = sp.WeibullPH.fit_from_df(
        df, x_col="x", c_col="c", Z_cols=["a", "b"]
    )
    assert model.concordance(df["x"], df["c"], df[["a", "b"]]) == (
        pytest.approx(model.concordance())
    )


def test_model_concordance_arguments():
    x, c, z = _data()
    model = sp.CoxPH.fit(x, z, c)
    with pytest.raises(ValueError, match="together"):
        model.concordance(x, c)
    restored = sp.from_dict(model.to_dict())
    with pytest.raises(ValueError, match="does not keep the data"):
        restored.concordance()
    assert restored.concordance(x, c, z) == pytest.approx(model.concordance())
