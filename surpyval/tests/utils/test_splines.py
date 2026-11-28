"""The shared natural cubic spline basis (surpyval/utils/splines.py).

Both forms are checked against SciPy's natural cubic spline and R's
``splines::ns`` (they must span exactly the curves those do), their
derivatives and antiderivative against SciPy's and against finite
differences, and the Durrleman-Simon form against the formula the
Royston-Parmar models always computed, to the last bit.
"""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from surpyval.utils.splines import (
    FORMS,
    natural_cubic_basis,
    natural_cubic_bends,
)

KNOTS = np.array([0.0, 1.5, 4.4, 6.0, 8.0])

# Rscript scripts/reference/reference_r_natural_spline.R
# R4.6.2, splines::ns
R_NS_X = [
    -2,
    0,
    0.69999999999999996,
    1.5,
    2.2000000000000002,
    3.1000000000000001,
    4.4000000000000004,
    5.9000000000000004,
    7.2999999999999998,
    8,
    9.5,
]
R_NS = [
    [
        -1.3941564123239147,
        0,
        -0.42688149422847133,
        1.1952681838397197,
        -0.76838668961124834,
    ],
    [
        -0.1996984610940864,
        0,
        -0.22599665106217906,
        0.63279062297410127,
        -0.40679397191192224,
    ],
    [
        0.19957209293650854,
        0.0086616161616161584,
        -0.15718043472698134,
        0.44010521723554774,
        -0.2829247825085664,
    ],
    [
        0.51126064361989787,
        0.085227272727272721,
        -0.090028327313962805,
        0.25207931647909582,
        -0.16205098916513302,
    ],
    [
        0.56368095914173155,
        0.24732291973671286,
        -0.045536298737561452,
        0.13882377027006448,
        -0.089243852316470029,
    ],
    [
        0.41118040235874054,
        0.49476384088453057,
        0.025520943995788799,
        0.063746779741352247,
        -0.040980072690869299,
    ],
    [
        0.092430398351667609,
        0.61766381766381762,
        0.26102151074393282,
        0.074199598976817396,
        -0.047699742199382611,
    ],
    [
        2.2566015222574873e-05,
        0.19779202279202271,
        0.62421719352274774,
        0.20534062950382762,
        -0.027372993966746275,
    ],
    [
        0,
        0.0073290598290598353,
        0.23503514674251438,
        0.34756077715514799,
        0.41007501627327775,
    ],
    [0, 0, -0.14900662251655628, 0.41721854304635758, 0.73178807947019875],
    [0, 0, -1.0057947019867548, 0.56622516556291347, 1.4395695364238414],
]


def _old_rcs(x, knots, deriv):
    """The Royston-Parmar basis as royston_parmar.py wrote it before the
    shared module."""
    kmin, kmax = knots[0], knots[-1]
    span = kmax - kmin
    if deriv == 0:
        cols = [np.ones_like(x), x]
    else:
        cols = [np.zeros_like(x), np.ones_like(x)]
    for kj in knots[1:-1]:
        lam = (kmax - kj) / span
        if deriv == 0:
            cols.append(
                np.maximum(x - kj, 0.0) ** 3
                - lam * np.maximum(x - kmin, 0.0) ** 3
                - (1.0 - lam) * np.maximum(x - kmax, 0.0) ** 3
            )
        else:
            cols.append(
                3.0 * np.maximum(x - kj, 0.0) ** 2
                - 3.0 * lam * np.maximum(x - kmin, 0.0) ** 2
                - 3.0 * (1.0 - lam) * np.maximum(x - kmax, 0.0) ** 2
            )
    return np.column_stack(cols)


def test_durrleman_simon_is_the_royston_parmar_basis_to_the_bit():
    rng = np.random.default_rng(0)
    for _ in range(100):
        knots = np.sort(rng.normal(size=rng.integers(2, 8))) * 3.0
        x = rng.normal(size=400) * 4.0
        for deriv in (0, 1):
            np.testing.assert_array_equal(
                natural_cubic_basis(x, knots, deriv), _old_rcs(x, knots, deriv)
            )


@pytest.mark.parametrize("form", FORMS)
def test_spans_scipys_natural_cubic_spline(form):
    # A natural cubic spline through any values at the knots is in the
    # span: solve for its coefficients at the knots, then it must agree
    # everywhere between the boundary knots, with its derivatives and
    # integral. (SciPy extrapolates the end cubics, not a line, so the
    # comparison stops at the boundary knots.)
    rng = np.random.default_rng(1)
    x = np.linspace(KNOTS[0], KNOTS[-1], 401)
    for _ in range(20):
        y = rng.normal(size=KNOTS.size)
        cs = CubicSpline(KNOTS, y, bc_type="natural")
        coef = np.linalg.solve(natural_cubic_basis(KNOTS, KNOTS, form=form), y)
        for deriv in (0, 1, 2):
            np.testing.assert_allclose(
                natural_cubic_basis(x, KNOTS, deriv, form) @ coef,
                cs(x, deriv),
                atol=1e-11,
            )
        area = (
            natural_cubic_basis(x, KNOTS, -1, form)
            - natural_cubic_basis(np.zeros_like(x), KNOTS, -1, form)
        ) @ coef
        np.testing.assert_allclose(
            area, [cs.integrate(0.0, v) for v in x], atol=1e-11
        )


@pytest.mark.parametrize("form", FORMS)
def test_spans_rs_ns(form):
    # each column of R's ns() is an exact combination of ours
    ours = natural_cubic_basis(R_NS_X, KNOTS, form=form)
    r = np.array(R_NS)
    combination, *_ = np.linalg.lstsq(ours, r, rcond=None)
    np.testing.assert_allclose(ours @ combination, r, atol=1e-12)


@pytest.mark.parametrize("form", FORMS)
def test_linear_outside_the_boundary_knots(form):
    lo = np.array([-9.0, -5.0, -1.0, 0.0])
    hi = np.array([8.0, 9.0, 13.0, 17.0])
    for x in (lo, hi):
        second = natural_cubic_basis(x, KNOTS, 2, form)
        np.testing.assert_allclose(second, 0.0, atol=1e-12)
        b = natural_cubic_basis(x, KNOTS, form=form)
        np.testing.assert_allclose(
            np.diff(b, axis=0) / np.diff(x)[:, None],
            natural_cubic_basis(x[:-1], KNOTS, 1, form),
            atol=1e-9,
        )


@pytest.mark.parametrize("form", FORMS)
def test_derivatives_are_the_basis_derivatives(form):
    x = np.linspace(-2.0, 10.0, 97) + 0.013
    h = 1e-5
    for deriv in (-1, 0, 1):
        up = natural_cubic_basis(x + h, KNOTS, deriv, form)
        down = natural_cubic_basis(x - h, KNOTS, deriv, form)
        np.testing.assert_allclose(
            (up - down) / (2 * h),
            natural_cubic_basis(x, KNOTS, deriv + 1, form),
            rtol=1e-6,
            atol=1e-6,
        )


def test_esl_bend_is_zero_up_to_its_knot():
    x = np.linspace(-3.0, 12.0, 301)
    bends = natural_cubic_bends(x, KNOTS, form="esl")
    for start, column in zip(KNOTS[:-2], bends):
        assert np.all(column[x <= start] == 0.0)
        assert np.all(column[x > start + 1e-9] != 0.0)


def test_bends_keep_the_shape_of_x():
    x = np.linspace(0.0, 8.0, 12).reshape(3, 4)
    for form in FORMS:
        bends = natural_cubic_bends(x, KNOTS, form=form)
        assert [b.shape for b in bends] == [(3, 4)] * 3
        flat = natural_cubic_basis(x.ravel(), KNOTS, form=form)[:, 2:]
        np.testing.assert_array_equal(
            np.column_stack([b.ravel() for b in bends]), flat
        )


def test_no_interior_knots_is_a_line():
    out = natural_cubic_basis([1.0, 2.0], [0.0, 3.0])
    np.testing.assert_array_equal(out, [[1.0, 1.0], [1.0, 2.0]])


@pytest.mark.parametrize(
    "kwargs, word", [({"deriv": 3}, "deriv"), ({"form": "bs"}, "form")]
)
def test_refusals_name_the_argument(kwargs, word):
    with pytest.raises(ValueError, match=word):
        natural_cubic_basis([1.0], KNOTS, **kwargs)
