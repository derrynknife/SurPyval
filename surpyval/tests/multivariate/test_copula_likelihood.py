"""The joint log-likelihood of a fitted copula model.

``CopulaModel`` reports the full censored/truncated joint log-likelihood
(``log_likelihood``/``neg_ll``) and ``aic``/``bic``, counting only the
parameters the fit estimated, and keeps them through serialisation.
"""

import json

import numpy as np
import pytest

import surpyval as surv
from surpyval import LogNormal, Weibull
from surpyval.multivariate import (
    AMH,
    Clayton,
    Frank,
    Gaussian,
    Gumbel,
    Independence,
    Joe,
    StudentT,
)

MARGINS = [Weibull.from_params([10.0, 2.0]), LogNormal.from_params([2.5, 0.5])]


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
        k = len(fam.parameter_names) + 4
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
    assert "_neg_ll" not in m.to_dict()


def test_likelihood_survives_serialisation(clayton_data):
    m = Clayton.fit(clayton_data, margins=[Weibull, LogNormal])
    restored = surv.from_dict(json.loads(json.dumps(m.to_dict())))
    assert restored.data is None
    assert restored.neg_ll() == m.neg_ll()
    assert restored.aic() == m.aic()
    assert restored.bic() == m.bic()
    # a dict written before the likelihood was stored has none
    old = m.to_dict()
    for key in ["_neg_ll", "k", "ic_n"]:
        del old[key]
    with pytest.raises(ValueError):
        surv.from_dict(old).neg_ll()


def test_likelihood_fields_are_bson_native(clayton_data):
    bson = pytest.importorskip("bson")
    m = Clayton.fit(clayton_data[:200], margins=[Weibull, LogNormal])
    restored = surv.from_dict(bson.decode(bson.encode(m.to_dict())))
    assert restored.aic() == m.aic()


# -- #619: quadrants and h-function complements evaluated directly ----------

NEAR_ONE = 1 - 2.0**-30
NEAR_ZERO = 2.0**-30
QUADRANTS = ("_survival", "_below_above", "_above_below")
COMPLEMENTS = ("_du_upper", "_dv_upper")

# (family, rotation, theta, u, v, [P(U > u, V > v), P(U <= u, V > v),
# P(U > u, V <= v), P(V > v | U = u), P(U > u | V = v)]) by mpmath at 400
# digits from the closed forms of C and dC/du, at the exact float inputs
# (1 - u is exact for these u). As 1 - u - v + C and 1 - dC/du, most of the
# small ones were 0 or noise.
CLOSED_FORM_REFERENCE = [
    (
        "Clayton",
        0,
        3.0,
        NEAR_ONE,
        NEAR_ONE,
        [
            3.4694469422600914e-18,
            9.313225711460316e-10,
            9.313225711460316e-10,
            3.7252902828494028e-09,
            3.7252902828494028e-09,
        ],
    ),
    (
        "Clayton",
        0,
        3.0,
        NEAR_ZERO,
        NEAR_ONE,
        [
            9.313225746154785e-10,
            7.006492334674694e-46,
            0.9999999981373549,
            3.00926554371025e-36,
            1.0,
        ],
    ),
    (
        "Clayton",
        90,
        3.0,
        NEAR_ONE,
        NEAR_ONE,
        [
            7.006492334674694e-46,
            9.313225746154785e-10,
            9.313225746154785e-10,
            3.00926554371025e-36,
            7.52316387328861e-37,
        ],
    ),
    (
        "Clayton",
        90,
        3.0,
        NEAR_ZERO,
        NEAR_ONE,
        [
            9.313225711460316e-10,
            3.4694469422600914e-18,
            0.9999999981373549,
            3.7252902828494028e-09,
            0.9999999962747097,
        ],
    ),
    (
        "Clayton",
        180,
        3.0,
        NEAR_ONE,
        NEAR_ONE,
        [
            7.391912173331713e-10,
            1.9213135728230724e-10,
            1.9213135728230724e-10,
            0.3968502629920499,
            0.3968502629920499,
        ],
    ),
    (
        "Clayton",
        180,
        3.0,
        NEAR_ZERO,
        NEAR_ONE,
        [
            9.313225746154785e-10,
            7.006492334674694e-46,
            0.9999999981373549,
            7.52316387328861e-37,
            1.0,
        ],
    ),
    (
        "Gumbel",
        0,
        1.0,
        NEAR_ONE,
        NEAR_ONE,
        [
            8.673617379884035e-19,
            9.313225737481168e-10,
            9.313225737481168e-10,
            9.313225746154785e-10,
            9.313225746154785e-10,
        ],
    ),
    (
        "Gumbel",
        0,
        1.0,
        NEAR_ZERO,
        NEAR_ONE,
        [
            9.313225737481168e-10,
            8.673617379884035e-19,
            0.9999999981373549,
            9.313225746154785e-10,
            0.9999999990686774,
        ],
    ),
    (
        "Gumbel",
        0,
        2.5,
        NEAR_ONE,
        NEAR_ONE,
        [
            6.33757644727291e-10,
            2.9756492988818756e-10,
            2.9756492988818756e-10,
            0.3402460448098725,
            0.3402460448098725,
        ],
    ),
    (
        "Gumbel",
        0,
        2.5,
        NEAR_ZERO,
        NEAR_ONE,
        [
            9.313225746154785e-10,
            1.0398971153094499e-34,
            0.9999999981373549,
            1.1971253207669258e-25,
            1.0,
        ],
    ),
    (
        "Gumbel",
        270,
        2.5,
        NEAR_ONE,
        NEAR_ONE,
        [
            1.0398971153094499e-34,
            9.313225746154785e-10,
            9.313225746154785e-10,
            2.791452564711639e-25,
            1.1971253207669258e-25,
        ],
    ),
    (
        "Gumbel",
        270,
        2.5,
        NEAR_ZERO,
        NEAR_ONE,
        [
            9.301101459935729e-10,
            1.2124286219056784e-12,
            0.9999999981385673,
            0.0008588910016019665,
            0.999141108998398,
        ],
    ),
    (
        "Joe",
        0,
        1.0,
        NEAR_ONE,
        NEAR_ONE,
        [
            8.673617379884035e-19,
            9.313225737481168e-10,
            9.313225737481168e-10,
            9.313225746154785e-10,
            9.313225746154785e-10,
        ],
    ),
    (
        "Joe",
        0,
        1.0,
        NEAR_ZERO,
        NEAR_ONE,
        [
            9.313225737481168e-10,
            8.673617379884035e-19,
            0.9999999981373549,
            9.313225746154785e-10,
            0.9999999990686774,
        ],
    ),
    (
        "Joe",
        90,
        4.0,
        NEAR_ONE,
        NEAR_ONE,
        [
            7.006492331412042e-46,
            9.313225746154785e-10,
            9.313225746154785e-10,
            7.523163866282117e-37,
            3.0092655423089514e-36,
        ],
    ),
    (
        "Joe",
        90,
        4.0,
        NEAR_ZERO,
        NEAR_ONE,
        [
            1.762128574799011e-10,
            7.551097171355774e-10,
            0.9999999988924646,
            0.4053964424986395,
            0.5946035575013605,
        ],
    ),
    (
        "AMH",
        0,
        -1.0,
        NEAR_ONE,
        NEAR_ONE,
        [
            1.6155871338926322e-27,
            9.313225746154785e-10,
            9.313225746154785e-10,
            2.6020852139652106e-18,
            2.6020852139652106e-18,
        ],
    ),
    (
        "AMH",
        0,
        -1.0,
        NEAR_ZERO,
        NEAR_ONE,
        [
            9.31322572880755e-10,
            1.7347234735534264e-18,
            0.9999999981373549,
            1.86264514576151e-09,
            0.9999999981373549,
        ],
    ),
    (
        "AMH",
        0,
        0.9,
        NEAR_ONE,
        NEAR_ONE,
        [
            1.6479873007239383e-18,
            9.313225729674912e-10,
            9.313225729674912e-10,
            1.7695128894275326e-09,
            1.7695128894275326e-09,
        ],
    ),
    (
        "AMH",
        0,
        0.9,
        NEAR_ZERO,
        NEAR_ONE,
        [
            9.313225745287423e-10,
            8.673617459855596e-20,
            0.9999999981373549,
            9.313225910086152e-11,
            0.9999999999068677,
        ],
    ),
    (
        "Frank",
        0,
        -20.0,
        NEAR_ONE,
        NEAR_ONE,
        [
            3.575531650407989e-26,
            9.313225746154785e-10,
            9.313225746154785e-10,
            3.839197911834121e-17,
            3.839197911834121e-17,
        ],
    ),
    (
        "Frank",
        0,
        -20.0,
        NEAR_ZERO,
        NEAR_ONE,
        [
            9.313225572682441e-10,
            1.7347234472405965e-17,
            0.9999999981373549,
            1.8626451010284518e-08,
            0.999999981373549,
        ],
    ),
    (
        "Frank",
        0,
        8.0,
        NEAR_ONE,
        NEAR_ONE,
        [
            6.941222372906512e-18,
            9.313225676742561e-10,
            9.313225676742561e-10,
            7.453080743709357e-09,
            7.453080743709357e-09,
        ],
    ),
    (
        "Frank",
        0,
        8.0,
        NEAR_ZERO,
        NEAR_ONE,
        [
            9.3132257461315e-10,
            2.32852073276859e-21,
            0.9999999981373549,
            2.500230108138845e-12,
            0.9999999999974998,
        ],
    ),
]

_FAMILIES = {
    "Clayton": Clayton,
    "Gumbel": Gumbel,
    "Joe": Joe,
    "AMH": AMH,
    "Frank": Frank,
}


@pytest.mark.parametrize(
    "name, rotation, theta, u, v, expected", CLOSED_FORM_REFERENCE
)
def test_619_quadrants_against_mpmath(name, rotation, theta, u, v, expected):
    family = _FAMILIES[name]
    if rotation:
        family = family.rotated(rotation)
    for method, value in zip(QUADRANTS + COMPLEMENTS, expected):
        got = float(
            getattr(family, method)(np.array([u]), np.array([v]), theta)[0]
        )
        assert got == pytest.approx(value, rel=1e-12, abs=0), method


# The elliptical copulas' upper quadrant under negative dependence: mpmath
# (30-40 digits) quadrature of the bivariate normal and t densities on a
# mesh graded towards the corner. The old 1 - u - v + C was 0 for every one
# of these but the t copula's with nu = 4.
ELLIPTICAL_REFERENCE = [
    (
        Gaussian,
        (-0.9,),
        1 - 2.0**-33,
        1 - 2.0**-33,
        3.1774178254911699564e-179,
    ),
    (Gaussian, (-0.5,), 1 - 2.0**-33, 1 - 2.0**-33, 1.4406772915482014622e-38),
    (StudentT, (-0.9, 50.0), 0.999, 0.999, 1.6890417041391504e-21),
    (StudentT, (-0.9, 50.0), 1 - 2.0**-20, 0.99, 8.078413217698636e-25),
    (StudentT, (-0.9, 4.0), 0.999, 0.999, 2.1781274448685645e-07),
    (StudentT, (-0.99, 1.5), 0.999, 0.999, 6.095137265555985e-07),
]


@pytest.mark.parametrize(
    "family, params, u, v, expected", ELLIPTICAL_REFERENCE
)
def test_619_elliptical_upper_quadrant(family, params, u, v, expected):
    got = float(family._survival(np.array([u]), np.array([v]), *params)[0])
    assert got == pytest.approx(expected, rel=1e-11, abs=0)


def test_619_gaussian_cdf_keeps_small_values():
    # scipy's bivariate normal CDF is accurate in absolute terms only: these
    # were 0 (mpmath as above)
    for u, rho, expected in [
        (0.01, -0.9, 2.0590500692148830324e-27),
        (0.001, -0.9, 1.2663046989146504537e-45),
        (2.0**-33, -0.9, 3.1774178254911699564e-179),
    ]:
        got = float(Gaussian.cdf(u, u, rho))
        assert got == pytest.approx(expected, rel=1e-12, abs=0)


def _row_loglik(family, params, u, v, c):
    """The copula likelihood of one row on uniform margins."""
    from surpyval import Uniform

    margin = Uniform.from_params([0.0, 1.0])
    ninf, inf = np.array([-np.inf]), np.array([np.inf])
    dims = [
        family._prepare_dim(
            margin,
            np.array([x]),
            np.array([k]),
            np.array([x]),
            np.array([x]),
            ninf,
            inf,
        )
        for x, k in zip((u, v), c)
    ]
    return float(family._pair_loglik(params, *dims)[0])


def test_619_doubly_right_censored_row_under_negative_dependence():
    # The #550 data's worst row, both series censored at their 99.9th
    # percentile, under the t copula with rho = -0.9 and nu = 50: it was
    # at the log floor (-690.8) and moved by 654 when C moved by one ulp.
    ll = _row_loglik(StudentT, [-0.9, 50.0], 0.999, 0.999, (1, 1))
    assert ll == pytest.approx(np.log(1.6890417041391504e-21), rel=1e-12)
    # One series observed, the other right censored: 1 - dC/du was 0
    ll = _row_loglik(StudentT, [-0.9, 50.0], 0.999, 0.999, (0, 1))
    assert ll == pytest.approx(np.log(3.6440178570741460851e-18), rel=1e-12)
    ll = _row_loglik(StudentT, [-0.9, 50.0], 0.999, 0.999, (1, 0))
    assert ll == pytest.approx(np.log(3.6440178570741460851e-18), rel=1e-12)


def test_619_left_truncated_rows_use_the_upper_quadrant():
    # Truncated on the left in both series: the window's mass is the upper
    # quadrant at the truncation points, which was 1 - u - v + C too
    mass = Gaussian._trunc_logmass(
        [-0.9],
        {"ul": np.array([1 - 2.0**-33]), "ur": np.array([1.0])},
        {"ul": np.array([1 - 2.0**-33]), "ur": np.array([1.0])},
    )
    assert float(mass[0]) == pytest.approx(
        np.log(3.1774178254911699564e-179), rel=1e-12
    )


@pytest.mark.parametrize(
    "family, params",
    [
        (Clayton, [2.0]),
        (Clayton.rotated(90), [2.0]),
        (Clayton.rotated(180), [2.0]),
        (Gumbel.rotated(270), [3.0]),
        (Joe, [2.0]),
        (Frank, [-5.0]),
        (AMH, [0.5]),
        (Gaussian, [-0.6]),
        (StudentT, [0.4, 5.0]),
    ],
)
def test_619_quadrants_partition_the_square(family, params):
    # The four quadrants sum to 1, each margin's two to its probability,
    # and each h-function and its complement to 1
    rng = np.random.default_rng(0)
    u, v = rng.uniform(0.01, 0.99, (2, 50))
    q = [
        family.cdf(u, v, *params),
        family._below_above(u, v, *params),
        family._above_below(u, v, *params),
        family._survival(u, v, *params),
    ]
    q = [np.asarray(x, dtype=float) for x in q]
    np.testing.assert_allclose(q[0] + q[1] + q[2] + q[3], 1.0, atol=1e-12)
    np.testing.assert_allclose(q[0] + q[1], u, atol=1e-12)
    np.testing.assert_allclose(q[0] + q[2], v, atol=1e-12)
    np.testing.assert_allclose(
        family.du(u, v, *params) + family._du_upper(u, v, *params),
        1.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        family.dv(u, v, *params) + family._dv_upper(u, v, *params),
        1.0,
        atol=1e-12,
    )


def test_619_model_sf_is_the_upper_quadrant():
    margins = [
        Weibull.from_params([10.0, 2.0]),
        LogNormal.from_params([2.5, 0.5]),
    ]
    model = Gaussian.from_params([-0.9], margins)
    x = [[float(margins[0].qf(0.999)), float(margins[1].qf(0.999))]]
    u = margins[0].ff(x[0][0])
    v = margins[1].ff(x[0][1])
    expected = Gaussian._survival(np.atleast_1d(u), np.atleast_1d(v), -0.9)
    assert float(model.sf(x)[0]) == pytest.approx(
        float(expected[0]), rel=1e-14
    )
    assert float(model.sf(x)[0]) > 0
