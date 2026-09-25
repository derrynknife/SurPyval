"""Regression tests for the second docs-review bug-fix round (copulas).

- The base ``Copula`` starting value respects the family's bounds (a fixed 1
  sat on the bound of a ``(-1, 1)`` family, the bounds transform made it
  infinite and the fit silently returned it), and ``fit`` takes a public
  ``init``.
- ``CopulaModel`` reports the full censored/truncated joint log-likelihood
  (``log_likelihood``/``neg_ll``) and ``aic``/``bic``.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval import LogNormal, Weibull
from surpyval.multivariate import (
    Clayton,
    Copula,
    Frank,
    Gaussian,
    Gumbel,
    Independence,
)

MARGINS = [Weibull.from_params([10.0, 2.0]), LogNormal.from_params([2.5, 0.5])]


class AliMikhailHaq(Copula):
    name = "Ali-Mikhail-Haq"
    bounds = ((-1, 1),)
    param_names = ("theta",)

    def cdf(self, u, v, theta):
        return u * v / (1 - theta * (1 - u) * (1 - v))


AMH = AliMikhailHaq()


@pytest.fixture(scope="module")
def amh_data():
    return AMH.from_params(0.6, margins=MARGINS).random(400, random_state=0)


# -- starting value ---------------------------------------------------------


def test_default_start_is_strictly_inside_the_bounds(amh_data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the old start warned (arctanh(1))
        model = AMH.fit(amh_data, margins=[Weibull, LogNormal])
    theta = model.params[0]
    # the old fit returned exactly the start, theta = 1
    assert -1 < theta < 1
    assert theta != 1.0
    assert theta == pytest.approx(0.6, abs=0.3)


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (((0, None),), [1.0]),  # 1 inside: the historical default is kept
        (((None, None),), [1.0]),
        (((-1, 1),), [0.0]),
        (((1, None),), [2.0]),
        (((None, 1),), [0.0]),
        (((2, 3),), [2.5]),
        (((0, None), (-1, 1)), [1.0, 0.0]),
    ],
)
def test_base_init_theta(bounds, expected):
    class Fam(Copula):
        pass

    fam = Fam()
    fam.bounds = bounds
    fam.param_names = tuple("p%d" % i for i in range(len(bounds)))
    np.testing.assert_array_equal(fam._init_theta([]), expected)


def test_public_init_argument(amh_data):
    a = AMH.fit(amh_data, margins=[Weibull, LogNormal], init=0.3)
    b = AMH.fit(amh_data, margins=[Weibull, LogNormal], init=[0.3])
    assert a.params[0] == b.params[0]
    assert a.params[0] == pytest.approx(
        AMH.fit(amh_data, margins=[Weibull, LogNormal]).params[0], abs=1e-3
    )
    # also used by the built-in families, and by MLE (through its IFM start)
    data = Clayton.from_params(2.0, margins=MARGINS).random(
        300, random_state=1
    )
    ifm = Clayton.fit(data, margins=[Weibull, LogNormal], init=5.0)
    ref = Clayton.fit(data, margins=[Weibull, LogNormal])
    assert ifm.params[0] == pytest.approx(ref.params[0], rel=1e-3)
    mle = Clayton.fit(data, margins=[Weibull, LogNormal], how="MLE", init=5.0)
    assert mle.params[0] == pytest.approx(ref.params[0], rel=0.2)


@pytest.mark.parametrize("bad", [1.0, -1.0, 2.0, np.nan, [0.1, 0.2]])
def test_init_outside_bounds_raises(amh_data, bad):
    with pytest.raises(ValueError, match="init"):
        AMH.fit(amh_data, margins=[Weibull, LogNormal], init=bad)


def test_subclass_start_on_bound_raises_not_silent(amh_data):
    class Stuck(AliMikhailHaq):
        def _init_theta(self, dims):
            return np.array([1.0])  # on the bound

    with pytest.raises(ValueError, match="strictly inside"):
        Stuck().fit(amh_data, margins=[Weibull, LogNormal])


def test_independence_accepts_empty_init():
    data = Independence.from_params([], margins=MARGINS).random(
        100, random_state=0
    )
    m = Independence.fit(data, margins=[Weibull, LogNormal], init=[])
    assert m.params.size == 0
    with pytest.raises(ValueError, match="init"):
        Independence.fit(data, margins=[Weibull, LogNormal], init=[0.5])


# -- log-likelihood and information criteria --------------------------------


@pytest.fixture(scope="module")
def clayton_data():
    return Clayton.from_params(2.0, margins=MARGINS).random(
        500, random_state=3
    )


def test_complete_data_loglik_is_sum_log_pdf(clayton_data):
    for fam in [Independence, Clayton, Gumbel, Frank, Gaussian]:
        m = fam.fit(clayton_data, margins=[Weibull, LogNormal])
        expected = np.sum(np.log(m.pdf(clayton_data)))
        assert m.log_likelihood == pytest.approx(expected, rel=1e-8)
        assert m.neg_ll() == pytest.approx(-expected, rel=1e-8)
        k = len(fam.param_names) + 4
        assert m.k == k
        assert m.aic() == pytest.approx(2 * k - 2 * expected)
        assert m.bic() == pytest.approx(
            k * np.log(len(clayton_data)) - 2 * expected
        )


def test_censored_loglik_uses_the_joint_probabilities(clayton_data):
    # rows 0..149 both right censored: log S(x1, x2); rows 150..349 series
    # 1 observed and series 2 right censored: log f1 (1 - dC/du); the rest
    # observed: log pdf
    x = clayton_data.copy()
    c = np.zeros_like(x, dtype=int)
    c[:150] = 1
    c[150:350, 1] = 1
    m = Clayton.fit(x, c=c, margins=[Weibull, LogNormal])
    F1, F2 = m.margins
    th = m.params[0]
    both = np.log(m.sf(x[:150]))
    u, v = F1.ff(x[150:350, 0]), F2.ff(x[150:350, 1])
    one = np.log(F1.df(x[150:350, 0]) * (1 - Clayton.du(u, v, th)))
    obs = np.log(m.pdf(x[350:]))
    expected = both.sum() + one.sum() + obs.sum()
    assert m.log_likelihood == pytest.approx(expected, rel=1e-8)
    # and it is not the complete-data sum of log pdf the docs used to use
    assert m.log_likelihood != pytest.approx(np.sum(np.log(m.pdf(x))))


def test_truncated_and_counted_loglik(clayton_data):
    field = clayton_data[clayton_data[:, 0] > 3.0][:200]
    t = np.empty((len(field), 2, 2))
    t[..., 0], t[..., 1] = -np.inf, np.inf
    t[:, 0, 0] = 3.0
    m = Clayton.fit(field, t=t, margins=[Weibull, LogNormal])
    F1 = m.margins[0]
    expected = np.sum(np.log(m.pdf(field))) - len(field) * np.log(F1.sf(3.0))
    assert m.log_likelihood == pytest.approx(expected, rel=1e-8)

    # counts multiply each row's contribution
    rows = np.ceil(clayton_data[:200])
    uniq, counts = np.unique(rows, axis=0, return_counts=True)
    by_count = Clayton.fit(uniq, n=counts, margins=[Weibull, LogNormal])
    assert by_count.log_likelihood == pytest.approx(
        np.sum(counts * np.log(by_count.pdf(uniq))), rel=1e-8
    )
    assert by_count.bic() == pytest.approx(
        by_count.k * np.log(200) + 2 * by_count.neg_ll()
    )


def test_k_counts_only_the_parameters_the_fit_estimated(clayton_data):
    m1 = Weibull.fit(clayton_data[:, 0])
    ifm = Clayton.fit(clayton_data, margins=[m1, LogNormal])
    assert ifm.k == 1 + 2  # the pre-fitted margin is used as it is
    mle = Clayton.fit(clayton_data[:200], margins=[m1, LogNormal], how="MLE")
    assert mle.k == 1 + 4  # MLE re-estimates every margin


def test_mle_loglik_at_least_ifm(clayton_data):
    small = clayton_data[:200]
    ifm = Clayton.fit(small, margins=[Weibull, LogNormal])
    mle = Clayton.fit(small, margins=[Weibull, LogNormal], how="MLE")
    assert mle.log_likelihood >= ifm.log_likelihood - 1e-6


def test_from_params_model_has_no_likelihood():
    m = Clayton.from_params(2.0, margins=MARGINS)
    for call in [m.neg_ll, m.aic, m.bic, lambda: m.log_likelihood]:
        with pytest.raises(ValueError, match="from_params"):
            call()
    # and it serialises without likelihood fields
    assert "neg_ll" not in m.to_dict()


def test_likelihood_survives_serialisation(clayton_data):
    m = Clayton.fit(clayton_data, margins=[Weibull, LogNormal])
    restored = surv.from_dict(json.loads(json.dumps(m.to_dict())))
    assert restored.data is None
    assert restored.neg_ll() == m.neg_ll()
    assert restored.aic() == m.aic()
    assert restored.bic() == m.bic()
    # a dict written before the likelihood was stored has none
    old = m.to_dict()
    for key in ["neg_ll", "k", "n_obs"]:
        del old[key]
    with pytest.raises(ValueError):
        surv.from_dict(old).neg_ll()


def test_likelihood_fields_are_bson_native(clayton_data):
    bson = pytest.importorskip("bson")
    m = Clayton.fit(clayton_data[:200], margins=[Weibull, LogNormal])
    restored = surv.from_dict(bson.decode(bson.encode(m.to_dict())))
    assert restored.aic() == m.aic()
