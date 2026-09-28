"""Shape in, shape out (principle 7; #381, #435).

Every function of a model evaluated at query points returns the query's
shape: a numpy scalar for a scalar, a 1-D array for a 1-D query, the 2-D
shape for a 2-D query and an empty array of the query's shape for an
empty one, with the values of the flat query. A two-sided bound adds a
trailing ``[lower, upper]`` axis.

The conformance suite (``conformance/test_vectorisation.py``) checks the
rule over every registered model. The tests here pin, with numbers, the
cases that went wrong before ``surpyval.utils.shapes``: several gave a
right-looking shape with wrong values (a 2-D query read with its axes
transposed or its coordinates mixed), which a shape check alone would
not have caught.
"""

import numpy as np
import pytest

import surpyval as surv
from surpyval import StepSchedule
from surpyval.tests.conformance.registry import CASE_BY_NAME, fitted
from surpyval.utils.shapes import flatten_query, keeps_query_shape

GRID = np.array([[2.0, 4.0], [6.0, 7.0]])


def _km():
    return surv.KaplanMeier.fit(
        [1, 2, 3, 4, 5, 6, 7, 8], c=[0, 1, 0, 0, 1, 0, 0, 1]
    )


def _assert_shape_in_shape_out(f, grid, width=(), jump=False):
    """``f`` of ``grid`` has its shape (plus ``width``) and the values of
    the flat query; a scalar gives a scalar (the same value as in the
    array, to round-off, except for a step estimate's jumps, ``jump``,
    which depend on their neighbours in the query); an empty query is
    empty."""
    flat = np.asarray(f(grid.ravel()))
    got = f(grid)
    assert np.shape(got) == grid.shape + width
    np.testing.assert_array_equal(
        np.reshape(got, (-1,) + width), flat.reshape((-1,) + width)
    )
    one = f(float(grid.flat[1]))
    assert np.shape(one) == width
    if not width:
        assert isinstance(one, np.float64)
    if not jump:
        np.testing.assert_allclose(
            one, flat.reshape((-1,) + width)[1], rtol=1e-12
        )
    assert np.shape(f(np.array([]))) == (0,) + width


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------
def test_flatten_query_restores_each_shape():
    for q in (3.0, np.array(3.0), [1.0, 2.0], GRID, np.empty((2, 0))):
        flat, restore = flatten_query(q)
        assert flat.ndim == 1
        out = restore(flat * 2)
        assert np.shape(out) == np.shape(q)
    flat, restore = flatten_query(3.0)
    assert isinstance(restore(flat), np.float64)
    # A two-sided bound keeps its [lower, upper] axis.
    flat, restore = flatten_query(GRID)
    assert restore(np.zeros((4, 2))).shape == (2, 2, 2)
    # A grid (rows x points) restores its last axis.
    assert restore(np.zeros((3, 4)), axis=-1).shape == (3, 2, 2)
    # A result that is not one per point is returned as it is.
    flat, restore = flatten_query(3.0)
    assert restore(np.zeros(5)).shape == (5,)
    # Points of a copula are pairs.
    flat, restore = flatten_query(np.zeros((2, 3, 2)), point_ndim=1)
    assert flat.shape == (6, 2)
    assert restore(np.zeros(6)).shape == (2, 3)


def test_keeps_query_shape_takes_the_query_by_name_or_none():
    class Model:
        @keeps_query_shape
        def sf(self, x, scale=1.0):
            if x is None:
                return "default"
            return np.exp(-np.atleast_1d(x) / scale)

    m = Model()
    assert isinstance(m.sf(0.0), np.float64)
    assert m.sf(x=GRID, scale=2.0).shape == (2, 2)
    assert m.sf(None) == "default"


# ---------------------------------------------------------------------------
# Non-parametric estimates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fname", ["sf", "ff", "Hf", "hf", "df"])
def test_nonparametric_functions_keep_the_query_shape(fname):
    # sf of a (2, 2) query was (2, 2, 2, 2); hf and df raised, and hf([])
    # raised IndexError.
    jump = fname in ("hf", "df")
    _assert_shape_in_shape_out(getattr(_km(), fname), GRID, jump=jump)


def test_nonparametric_bounds_of_a_2d_query_are_the_flat_bounds():
    # The (2, 2) query came back (2, 2, 2) but transposed: at t = 2 the
    # "bound" was [0.724, 0.063], lower above upper; it is [0.387, 0.981].
    model = _km()
    got = model.cb(GRID)
    np.testing.assert_allclose(got[0, 0], [0.38700001, 0.98139297])
    _assert_shape_in_shape_out(model.cb, GRID, (2,))
    _assert_shape_in_shape_out(lambda q: model.cb(q, bound="lower"), GRID)
    _assert_shape_in_shape_out(model.R_cb, GRID, (2,))
    _assert_shape_in_shape_out(model.smoothed_hf, GRID)
    _assert_shape_in_shape_out(
        lambda q: model.band(q, method="nair"), GRID, (2,)
    )


def test_nonparametric_quantiles_keep_the_query_shape():
    model = _km()
    assert model.qf(0.5) == 6.0 and isinstance(model.qf(0.5), np.float64)
    assert model.median == 6.0
    assert model.qf([[0.2, 0.5]]).shape == (1, 2)
    assert model.quantile_cb(0.25).shape == (2,)
    np.testing.assert_array_equal(model.quantile_cb(0.25), [1.0, 6.0])


# ---------------------------------------------------------------------------
# Parametric bounds and the special distributions
# ---------------------------------------------------------------------------
def test_parametric_bounds_keep_the_query_shape():
    np.random.seed(1)
    model = surv.Weibull.fit(surv.Weibull.random(30, 10, 3))
    # cb(5) was (1, 2); cb of a (2, 2) query raised (Wald) or repeated
    # the first row's bound (likelihood ratio).
    _assert_shape_in_shape_out(model.cb, GRID, (2,))
    lr = model.cb(GRID, method="lr")
    assert lr.shape == (2, 2, 2)
    np.testing.assert_allclose(
        lr.reshape(-1, 2), model.cb(GRID.ravel(), method="lr"), rtol=1e-6
    )
    assert model.cb([]).shape == (0, 2)


def test_special_distributions_give_a_scalar_for_a_scalar():
    assert isinstance(surv.NeverOccurs.sf(3.0), np.float64)
    assert isinstance(surv.InstantlyOccurs.qf(0.5), np.float64)
    assert np.shape(surv.Bernoulli.fit([0, 1, 1, 0, 1]).sf(1)) == ()
    assert np.shape(surv.ExactEventTime.sf(4.0, 5.0)) == ()
    mixture = fitted(CASE_BY_NAME["MixtureModel"])
    assert isinstance(mixture.df(5.0), np.float64)
    # BetaGeometric.qf raised for a 2-D and for an empty query.
    q = surv.BetaGeometric.qf(np.array([[0.2, 0.5]]), 2.0, 3.0)
    np.testing.assert_array_equal(q, [[1.0, 2.0]])
    assert surv.BetaGeometric.qf(np.array([]), 2.0, 3.0).shape == (0,)


def test_royston_parmar_keeps_the_query_shape():
    # Every function of a 2-D query raised (a matmul with the basis).
    model = fitted(CASE_BY_NAME["RoystonParmar"])
    for fname in ("sf", "ff", "Hf", "hf", "df"):
        _assert_shape_in_shape_out(getattr(model, fname), GRID)
    _assert_shape_in_shape_out(model.cb, GRID, (2,))
    _assert_shape_in_shape_out(model.qf, GRID / 10)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def _ph_data():
    rng = np.random.default_rng(0)
    Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
    x = 100 * rng.weibull(2, 200) * np.exp(-0.25 * Z[:, 0])
    return x, Z


@pytest.mark.parametrize(
    "fitter", ["WeibullAFT", "WeibullPO", "WeibullPH", "CoxPH"]
)
def test_regression_scalar_query_gives_a_scalar(fitter):
    # AFT, PO and Cox gave (1,) for sf(20.0, z); every cb gave (1, 2).
    x, Z = _ph_data()
    model = getattr(surv, fitter).fit(x, Z)
    for fname in ("sf", "ff", "Hf", "hf", "df"):
        f = getattr(model, fname)
        _assert_shape_in_shape_out(lambda q: f(q, [1.0]), GRID * 10)
    # Several rows at one time: one value per row, as before.
    assert model.sf(20.0, [[1.0], [0.0]]).shape == (2,)
    schedule = StepSchedule.constant([1.0])
    _assert_shape_in_shape_out(lambda q: model.sf_tvc(q, schedule), GRID * 10)
    if hasattr(model, "cb"):
        assert model.cb(20.0, [1.0]).shape == (2,)
        assert np.shape(model.cb(20.0, [1.0], bound="lower")) == ()


def test_sf_tvc_given_nan_is_nan():
    # #435 item 1 (the same code as the shape fix): the parametric
    # families ignored given=nan and returned 0.936 at 20.
    x, Z = _ph_data()
    model = surv.WeibullPH.fit(x, Z)
    got = model.sf_tvc(
        [20.0, 50.0], StepSchedule.constant([1.0]), given=np.nan
    )
    assert np.isnan(got).all()


def test_additive_hazards_rate_of_a_2d_query():
    # hf and df of a 2-D query raised (the kernel broadcast).
    x, Z = _ph_data()
    model = surv.AdditiveHazards.fit(x / 100, Z)
    for fname in ("hf", "df", "sf"):
        f = getattr(model, fname)
        _assert_shape_in_shape_out(lambda q: f(q, [1.0]), GRID / 10)


def test_trees_keep_the_shape_of_x_and_add_the_rows_axis():
    # With one covariate vector the result is shaped like x (a scalar gave
    # (1,)); with a matrix it is the documented grid, (n_rows,) + x.shape.
    for name in ("SurvivalTree[weibull]", "RandomSurvivalForest"):
        case = CASE_BY_NAME[name]
        model = fitted(case)
        z = case.Z[1]
        _assert_shape_in_shape_out(lambda q: model.sf(q, z), GRID)
        grid = model.sf(GRID, case.Z[:3])
        assert grid.shape == (3, 2, 2)
        np.testing.assert_array_equal(grid[1], model.sf(GRID, case.Z[1]))


# ---------------------------------------------------------------------------
# Competing risks and recurrent events
# ---------------------------------------------------------------------------
def test_competing_risks_keep_the_query_shape():
    case = CASE_BY_NAME["CompetingRisksProportionalHazards[Cox]"]
    model = fitted(case)
    z = case.Z[1]
    # cif of a (2, 2) query came back flat, (4,).
    _assert_shape_in_shape_out(lambda q: model.cif(q, z, "a"), GRID)
    np_model = fitted(CASE_BY_NAME["CompetingRisks[Nelson-Aalen]"])
    _assert_shape_in_shape_out(lambda q: np_model.cif(q, "a"), GRID)
    fg = fitted(CASE_BY_NAME["FineGray"])
    _assert_shape_in_shape_out(lambda q: fg.cif(q, z), GRID)


def test_mcf_bounds_of_a_2d_query_are_the_flat_bounds():
    # mcf_cb of a (2, 2) query came back (2, 2, 2) with the axes
    # transposed (an upper bound of 1.0 below the MCF of 1.73 at t = 25).
    model = fitted(CASE_BY_NAME["NonParametricCounting"])
    q = np.array([[1.0, 5.0], [12.0, 25.0]])
    _assert_shape_in_shape_out(model.mcf, q)
    _assert_shape_in_shape_out(model.mcf_cb, q, (2,))
    _assert_shape_in_shape_out(
        lambda t: model.mcf_cb(t, interp="linear"), q, (2,)
    )


def test_recurrent_models_keep_the_query_shape():
    model = fitted(CASE_BY_NAME["CrowAMSAA"])
    _assert_shape_in_shape_out(model.cif_cb, GRID, (2,))
    renewal = fitted(CASE_BY_NAME["GeneralizedOneRenewal"])
    # mcf([]) raised ValueError (the max of an empty array).
    assert renewal.mcf([], random_state=1).shape == (0,)
    assert np.shape(renewal.mcf(5.0, random_state=1)) == ()


# ---------------------------------------------------------------------------
# Degradation and copulas
# ---------------------------------------------------------------------------
def test_degradation_models_keep_the_query_shape():
    wiener = fitted(CASE_BY_NAME["WienerProcess"])
    # qf of a (2, 2) query raised; qf([0.5]) was a float, not (1,).
    _assert_shape_in_shape_out(wiener.qf, GRID / 10)
    assert wiener.qf([0.5]).shape == (1,)
    _assert_shape_in_shape_out(wiener.sf, GRID * 50)
    induced = fitted(CASE_BY_NAME["InducedFailureDistribution"])
    # sf of a (2, 2) query raised (broadcast against the samples).
    _assert_shape_in_shape_out(induced.sf, GRID * 150)
    destructive = fitted(CASE_BY_NAME["DestructiveDegradation"])
    # cb of a (2, 2) query read it as one flat row: a (2, 2) "two-sided"
    # bound of [0.99657, 1.0] and [1.1e-6, 1.0] where the bounds at the
    # four times are [1, 1], [1, 1], [1, 1] and [0.9962, 0.9995].
    t = np.array([[5.0, 20.0], [40.0, 60.0]])
    got = destructive.cb(t, n_boot=10, random_state=1)
    flat = destructive.cb(t.ravel(), n_boot=10, random_state=1)
    assert got.shape == (2, 2, 2)
    np.testing.assert_array_equal(got.reshape(-1, 2), flat)
    assert destructive.cb(40.0, n_boot=10, random_state=1).shape == (2,)


def test_copula_points_keep_their_shape():
    from surpyval.multivariate import Clayton

    margins = [
        surv.Weibull.from_params([10, 2]),
        surv.Weibull.from_params([20, 3]),
    ]
    model = Clayton.from_params([2.0], margins)
    X = np.array([[[5.0, 15.0], [10.0, 20.0]], [[8.0, 12.0], [3.0, 30.0]]])
    # A (2, 2, 2) query of four points mixed their coordinates: sf gave
    # [[0.766, 0.076], [0.527, 0.019]]; it is [[0.624, 0.235], [0.516,
    # 0.034]].
    got = model.sf(X)
    np.testing.assert_allclose(
        got, [[0.62400795, 0.23542792], [0.5156836, 0.03419514]]
    )
    np.testing.assert_array_equal(got.ravel(), model.sf(X.reshape(-1, 2)))
    assert isinstance(model.cdf([5.0, 15.0]), np.float64)
    gaussian = fitted(CASE_BY_NAME["GaussianCopula"])
    # sf of an empty (0, 2) query raised in scipy.
    assert gaussian.sf(np.empty((0, 2))).shape == (0,)
