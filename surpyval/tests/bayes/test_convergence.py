"""The convergence statistics of a sampler's draws
(``surpyval.bayes.convergence``): against ArviZ's values, against the
definitions written out plainly, and in exact arithmetic."""

from fractions import Fraction

import numpy as np
import pytest

from surpyval.bayes import convergence as cv


def chains(m, n, phi, shift=0.0, digits=None):
    """The chains of ``scripts/reference/reference_arviz_convergence.py``:
    integer arithmetic and IEEE additions and multiplications only, so
    they are the same on every platform."""
    state = 12345
    out = np.empty((m, n))
    for k in range(m):
        x = 0.0
        for t in range(n):
            state = (1103515245 * state + 12345) % 2147483648
            x = phi * x + state / 2147483648 - 0.5
            out[k, t] = x + k * shift
    if digits is not None:
        out = np.round(out, digits)
    return out


CASES = {
    "mixed": (4, 200, 0.5, 0.0, None),
    "sticky": (4, 200, 0.95, 0.0, None),
    "apart": (4, 200, 0.5, 0.3, None),
    "odd_with_ties": (3, 101, 0.7, 0.0, 1),
    "one_chain": (1, 400, 0.3, 0.0, None),
}

# ArviZ 0.22.0: rhat(method="rank"), ess(method="bulk"), ess(method="tail")
# and ess(method="mean"), by scripts/reference/reference_arviz_convergence.py
# (ArviZ gives no R-hat for one chain).
ARVIZ = {
    "mixed": (
        1.0036877781779199,
        265.87134918022707,
        510.62145401768043,
        271.36390057572805,
    ),
    "sticky": (
        1.0931518058184024,
        31.058437889582574,
        53.73353171995386,
        31.697460802826303,
    ),
    "apart": (
        1.5218822374507328,
        7.6964151197086705,
        47.318681666055504,
        7.518841657819765,
    ),
    "odd_with_ties": (
        1.039650536807568,
        53.49655091182378,
        51.82334207675433,
        54.88691631956967,
    ),
    "one_chain": (
        None,
        182.5143289394168,
        255.73831688386733,
        192.5750505128532,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_agrees_with_arviz(name):
    x = chains(*CASES[name])
    rhat, bulk, tail, mean = ARVIZ[name]
    if rhat is not None:
        assert cv.rhat(x) == pytest.approx(rhat, rel=1e-10)
    assert cv.ess_bulk(x) == pytest.approx(bulk, rel=1e-10)
    assert cv.ess_tail(x) == pytest.approx(tail, rel=1e-10)
    # ArviZ's "mean" is of the split chains
    assert cv.ess(cv._split(x)) == pytest.approx(mean, rel=1e-10)


def test_autocovariance_is_the_plain_sum():
    x = chains(3, 37, 0.6)
    got = cv._autocovariance(x)
    for k in range(3):
        d = x[k] - x[k].mean()
        for lag in (0, 1, 5, 36):
            want = np.sum(d[: 37 - lag] * d[lag:]) / 37
            assert got[k, lag] == pytest.approx(want, abs=1e-13)


def test_split_rhat_of_a_short_chain_in_exact_arithmetic():
    """The ratio before ranks, worked by hand in fractions for two chains
    of four draws (four half chains of two): the squared value is
    ``((n - 1)/n W + B/n) / W`` with ``n = 2``."""
    x = np.array([[1.0, 2.0, 4.0, 7.0], [3.0, 3.0, 5.0, 9.0]])
    halves = [[1, 2], [4, 7], [3, 3], [5, 9]]
    means = [Fraction(sum(h), 2) for h in halves]
    variances = [
        sum((Fraction(v) - m) ** 2 for v in h) for h, m in zip(halves, means)
    ]  # divisor n - 1 = 1
    within = sum(variances, Fraction(0)) / 4
    grand = sum(means, Fraction(0)) / 4
    between = sum((m - grand) ** 2 for m in means) / 3
    squared = (Fraction(1, 2) * within + between) / within
    assert squared == Fraction(185, 78)
    got = cv._plain_rhat(cv._split(x))
    assert got**2 == pytest.approx(float(squared), rel=1e-14)


def test_normal_scores_of_ranks_by_hand():
    """Four draws, ranks 1 to 4 with a tie for 2 and 3: the scores are
    the normal quantiles of ``(r - 3/8) / (4 + 1/4)``."""
    from statistics import NormalDist

    z = cv._z_scale(np.array([[5.0, 1.0], [5.0, 9.0]]))
    q = NormalDist().inv_cdf
    want = [[q(2.125 / 4.25), q(0.625 / 4.25)], [0.0, q(3.625 / 4.25)]]
    assert z == pytest.approx(np.array(want), abs=1e-12)


def test_scale_does_not_matter_to_the_effective_sample_sizes():
    """Nor to R-hat before folding; the folded part (the draws'
    distances from their median) does depend on the scale."""
    x = np.exp(chains(4, 200, 0.5))
    for fn in (cv.ess_bulk, cv.ess_tail):
        assert fn(np.log(x)) == pytest.approx(fn(x), rel=1e-12)
    bulk = [cv._plain_rhat(cv._z_scale(cv._split(v))) for v in (x, np.log(x))]
    assert bulk[0] == pytest.approx(bulk[1], rel=1e-12)


def test_constant_and_non_finite_draws_give_nan():
    assert np.isnan(cv.rhat(np.ones((2, 10))))
    assert np.isnan(cv.ess_bulk(np.ones((2, 10))))
    x = chains(2, 10, 0.5)
    x[0, 0] = np.inf
    assert np.isnan(cv.ess_tail(x))


def test_shape_is_checked():
    with pytest.raises(ValueError, match="chains, draws"):
        cv.rhat(np.zeros((2, 3)))


def test_check_names_the_parameter_and_the_number():
    x = chains(4, 200, 0.5, shift=0.3)
    D = np.stack([x, chains(4, 200, 0.2)], axis=2)
    div = np.zeros((4, 200), dtype=bool)
    div[1, :3] = True
    depth = np.full((4, 200), 3)
    depth[0, :2] = 10
    s = cv.summarise(D, ["a", "b"], div, depth, 10)
    assert s["divergences"] == 3 and s["tree_depth_saturated"] == 2
    problems = cv.check(s, 4, 200)
    text = "; ".join(problems)
    assert "R-hat above 1.01: a 1.522" in text
    assert "bulk effective sample size below 400: a 8" in text
    assert "3 of the 800 kept iterations diverged" in text
    assert "2 of the 800 kept iterations reached the largest tree" in text
    assert "b " not in text.split("tail")[0]
    good = cv.summarise(chains(4, 2000, 0.2)[:, :, None], ["b"])
    assert cv.check(good, 4, 2000) == []
