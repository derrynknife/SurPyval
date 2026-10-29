"""The prior constructors of ``surpyval.prior``: the arguments each one
refuses, with a message that names the argument (principle 2), and a
density prior whose support leaves its parameter's range."""

import numpy as np
import pytest

import surpyval as sp
from surpyval import prior

REFUSED = {
    "around_ends_not_increasing": (
        lambda: prior.around(2, between=(3, 1)),
        "between",
    ),
    "around_mass_above_1": (
        lambda: prior.around(2, between=(1, 3), mass=1.5),
        "mass",
    ),
    "around_mass_0": (
        lambda: prior.around(2, between=(1, 3), mass=0),
        "mass",
    ),
    "between_ends_not_increasing": (lambda: prior.between(3, 1), "a < b"),
    "uniform_ends_not_increasing": (lambda: prior.uniform(3, 1), "a < b"),
    "fixed_nan": (lambda: prior.fixed(np.nan), "fixed"),
    "fixed_inf": (lambda: prior.fixed(np.inf), "fixed"),
    "mvn_cov_wrong_shape": (
        lambda: prior.MultivariateNormal(["a", "b"], [0, 0], np.eye(3)),
        "cov",
    ),
    "mvn_mean_wrong_length": (
        lambda: prior.MultivariateNormal(["a", "b"], [0, 0, 0], np.eye(2)),
        "mean",
    ),
    "mvn_cov_not_symmetric": (
        lambda: prior.MultivariateNormal(
            ["a", "b"], [0, 0], [[1.0, 0.5], [0.0, 1.0]]
        ),
        "cov must be a symmetric",
    ),
    "mvn_cov_not_positive_definite": (
        lambda: prior.MultivariateNormal(
            ["a", "b"], [0, 0], [[1.0, 2.0], [2.0, 1.0]]
        ),
        "positive-definite",
    ),
}


@pytest.mark.parametrize("case", list(REFUSED))
def test_constructor_refuses_naming_the_argument(case):
    """audit B24: each constructor's invalid arguments raise a ValueError
    naming the argument."""
    make, match = REFUSED[case]
    with pytest.raises(ValueError, match=match):
        make()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "prior.around(5, between=(1, 3)) is accepted: the stated centre "
        "5 lies outside the interval it is said almost surely to lie in, "
        "and the prior built ('about 5 (90% between 1 and 3)') has its "
        "median at sqrt(3) on a positive parameter. The docstring promises "
        "only that the summary says what was built where the median is "
        "not m; a centre outside the interval is a contradiction that "
        "principle 2 would refuse."
    ),
)
def test_around_refuses_a_centre_outside_its_interval():
    """audit B24: a centre outside ``between`` is refused, naming it."""
    with pytest.raises(ValueError, match="between"):
        prior.around(5, between=(1, 3))


def test_a_distribution_leaving_the_range_is_refused_until_truncated():
    """audit B24: a Normal prior on a positive parameter puts mass below
    0; the refusal points to ``truncate=True``, which is then taken.
    (``test_a_normal_prior_on_a_positive_parameter_is_refused`` in
    test_fit_bayes.py checks the refusal's opening words.)"""
    x = sp.Weibull.random(30, 100, 2, random_state=5)
    normal = sp.Normal.from_params([2, 1])
    with pytest.raises(ValueError, match="truncate=True"):
        sp.Weibull.fit_bayes(x, priors={"beta": normal})
    model = sp.Weibull.fit_bayes(
        x, priors={"beta": prior.from_distribution(normal, truncate=True)}
    )
    assert model.params[1] > 0
    assert model.posterior.weights.sum() == pytest.approx(1.0)
