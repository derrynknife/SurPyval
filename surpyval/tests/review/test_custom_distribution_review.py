"""Targeted review of ``parametric/distributions/custom_distribution.py``
(#399).

Each test pins a bug found by reading the module adversarially (fixed
under #437; they were strict expected failures until then).
"""

import warnings

import numpy as np
import pytest

import surpyval as surv


def _weibull_hf(x, *params):
    return (x / params[0]) ** params[1]


def _data():
    np.random.seed(0)
    return surv.Weibull.random(50, 10, 2)


def test_parameter_named_k_is_refused():
    # A fit exposes each parameter by name on the model, and 'k' overwrote
    # the model's parameter count: AIC 290.08 (k = 2.29) against 289.50
    # for the same fit with the parameter named 'shape'.
    with pytest.raises(ValueError, match=r"\['k'\].*Reserved: .*\bk\b"):
        surv.CustomDistribution(
            "review_weibull_k",
            _weibull_hf,
            ["lam", "k"],
            ((0, None), (0, None)),
            (0, np.inf),
        )
    x = _data()
    shape = surv.CustomDistribution(
        "review_weibull_shape",
        _weibull_hf,
        ["lam", "shape"],
        ((0, None), (0, None)),
        (0, np.inf),
    ).fit(x)
    assert shape.k == 2


# The names broke sf/bic/cb with AttributeError, IndexError or ValueError
# after the fit; the constructor refuses them instead.
@pytest.mark.parametrize("name", ["dist", "data", "lfp", "zi", "method"])
def test_parameter_names_that_collide_with_the_model_are_refused(name):
    with pytest.raises(ValueError, match="Reserved: "):
        surv.CustomDistribution(
            "review_collide_" + name,
            _weibull_hf,
            ["lam", name],
            ((0, None), (0, None)),
            (0, np.inf),
        )


def test_qf_outside_the_unit_interval_is_nan():
    # qf(-0.1) was the support's lower bound (0.0)
    dist = surv.CustomDistribution(
        "review_qf",
        _weibull_hf,
        ["lam", "shape"],
        ((0, None), (0, None)),
        (0, np.inf),
    )
    # One warning from every qf there (#611), not numpy's.
    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        got = dist.qf(np.array([-0.1, 1.5]), 10.0, 3.0)
    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        assert np.isnan(surv.Weibull.qf(np.array([-0.1]), 10.0, 3.0)).all()
    with np.errstate(all="ignore"):
        want = surv.Weibull.qf(np.array([0.0, 0.5, 1.0]), 10.0, 3.0)
    assert np.isnan(got).all()
    np.testing.assert_allclose(
        dist.qf(np.array([0.0, 0.5, 1.0]), 10.0, 3.0), want, rtol=1e-12
    )


@pytest.mark.parametrize(
    "tag, fun",
    [
        ("star", lambda t, *theta: (t / theta[0]) ** theta[1]),
        ("named", lambda x, lam, shape: (x / lam) ** shape),
        ("default", lambda x, lam, shape=2.0: (x / lam) ** shape),
    ],
)
def test_any_star_arg_name_or_named_parameters_are_accepted(tag, fun):
    # Only the exact text '(x, *params)' was accepted.
    dist = surv.CustomDistribution(
        "review_signature_" + tag,
        fun,
        ["lam", "shape"],
        ((0, None), (0, None)),
        (0, np.inf),
    )
    np.testing.assert_allclose(
        dist.sf(np.array([5.0, 10.0]), 10.0, 3.0),
        surv.Weibull.sf(np.array([5.0, 10.0]), 10.0, 3.0),
        rtol=1e-12,
    )
    x = _data()
    np.testing.assert_allclose(
        dist.fit(x).params, surv.Weibull.fit(x).params, rtol=1e-4
    )


@pytest.mark.parametrize(
    "fun",
    [
        lambda x, lam: x / lam,  # one parameter short
        lambda x, a, b, c: x,  # one too many
        lambda x, a, *rest: x,  # a star-argument after named ones
        lambda x, *p, scale: x,  # a keyword-only argument to fill
    ],
)
def test_a_signature_that_does_not_fit_the_parameters_is_refused(fun):
    with pytest.raises(ValueError, match="must take the time"):
        surv.CustomDistribution(
            "review_bad_signature",
            fun,
            ["lam", "shape"],
            ((0, None), (0, None)),
            (0, np.inf),
        )


def test_replacing_a_registered_name_warns():
    def args():
        return (
            ["lam", "shape"],
            ((0, None), (0, None)),
            (0, np.inf),
        )

    first = surv.CustomDistribution("review_twice", _weibull_hf, *args())
    model = first.fit(_data())
    saved = model.to_dict()
    with warnings.catch_warnings():
        # the very same definition again changes nothing
        warnings.simplefilter("error")
        surv.CustomDistribution("review_twice", _weibull_hf, *args())

    def other(x, *params):
        return 2 * (x / params[0]) ** params[1]

    with pytest.warns(UserWarning, match="replaces it in the registry"):
        second = surv.CustomDistribution("review_twice", other, *args())
    assert surv.from_dict(saved).dist is second
