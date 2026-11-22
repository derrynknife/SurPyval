"""The joint model's integral by Laplace's approximation
(``integration="laplace"``): the default beyond five varying path
parameters, an option below.

The checks:

- the stencil's derivatives on a function whose derivatives are known;
- a closed form: with no marker effect and a path linear in its
  parameters the integrand is exactly normal, so the approximation is
  exact, in two, three, four and six dimensions (the grid has no six);
- with a marker effect, against the 7-point grid and an 11-point one in
  two and three dimensions at the simulation's truth, read often and
  read seldom;
- its derivatives: the gradient by the held stencils against differences
  of the approximation with the stencils placed again at each point
  (and that the implicit-function term is what makes them agree), and
  the curvature the standard errors come from against second
  differences of the same;
- fits against the grid's in two and three dimensions (the recovery
  study's pumps and a quadratic's fleet), a six-parameter fleet
  recovered, twice the same;
- predictions: on the grid up to five varying parameters, Laplace's
  beyond, against the closed form with no marker effect;
- the default, the refusals (a grid beyond five, truncated items), the
  warning where many items have few readings, saving and restoring.

The bounds on the approximation's own error (it is not exact with a
marker effect) are a few times what was first measured, recorded beside
each.
"""

import copy
import functools
import pickle
import warnings

import numpy as np
import pytest
from autograd import value_and_grad
from scipy.stats import multivariate_normal

from surpyval import JointAFT, JointPH, Weibull
from surpyval.degradation import LinearPath, QuadraticPath
from surpyval.tests.univariate.joint import _fleet
from surpyval.tests.univariate.joint import _laplace_fleet as knots
from surpyval.tests.univariate.joint import test_joint_integration as grids
from surpyval.univariate.joint import JointModel
from surpyval.univariate.joint import _laplace as lap
from surpyval.univariate.joint import _likelihood as lik
from surpyval.univariate.joint import joint_ph
from surpyval.univariate.joint.joint_model import parameter_names
from surpyval.univariate.joint.joint_ph import likelihood_parts

KNOT_PATH = knots.KNOT_PATH


class LinePath(grids.PolynomialPath):
    """``y = a + b t`` as a path of one's own (the tests' polynomial
    fleets have one for each degree)."""

    def __init__(self):
        self.name = "Line"
        self.parameter_names = ["a", "b"]


LINE = LinePath()
PATHS = {2: LINE, 3: QuadraticPath, 4: grids.CUBIC}


def knots_at(integration="laplace", effect=knots.EFFECT):
    """A joint model of the bump path (every bump's height varying,
    uncorrelated) at the truth, without a fit."""
    names = list(KNOT_PATH.parameter_names)
    values = dict(zip(names, knots.MEANS))
    values.update({f"sd_{n}": s for n, s in zip(names, knots.SDS)})
    values.update(
        sd_noise=knots.NOISE,
        alpha=knots.ALPHA,
        beta=knots.BETA,
        marker_effect=effect,
    )
    out = JointModel()
    out.dist = Weibull
    out.path_model = KNOT_PATH
    out.varying = names
    out.parameter_names = parameter_names(names, names, [])
    out.params = np.array(
        [values.get(name, 0.0) for name in out.parameter_names]
    )
    out.feature_names = []
    out.aliased = np.array([], dtype=int)
    out.marker_reference = knots.CENTRE
    out._centre = knots.CENTRE
    out._zbar = np.zeros(0)
    out.integration = integration
    out.k = len(out.params)
    return out


def placed_again(model, data):
    """``f(u)``: the Laplace negative log-likelihood of ``data`` with the
    stencils placed at ``u`` itself (each item's mode found there), and
    the pieces it is made of."""
    parts, arrays, u = likelihood_parts(model, **data)

    def f(v):
        placed, _ = lik.place(v, parts, arrays, (None, None))
        return float(lik.neg_ll(v, parts, arrays, placed))

    return f, parts, arrays, u


# -- the stencil -------------------------------------------------------------


@pytest.mark.parametrize("q", [1, 2, 3, 6])
def test_the_stencil_gives_a_quartics_derivatives(q):
    # g(z) = a'z - z'Az/2 + sum z_i^3 + sum z_i^4: its gradient at 0 is
    # a, which a central difference of step h is off from by the cubic's
    # h^2, and its second derivative -A, off by the quartic's 2 h^2 on
    # the diagonal (the cubic's part cancels there)
    rng = np.random.default_rng(50 + q)
    a = rng.standard_normal(q)
    root = rng.standard_normal((q, q))
    A = root @ root.T + q * np.eye(q)
    Z, Cg, Ch = lap.stencil(q)
    assert Z.shape == (q * q + q + 1, q)
    z = lap.STEP * Z
    g = z @ a - 0.5 * np.einsum("ki,ij,kj->k", z, A, z)
    g = g + np.sum(z**3, axis=1) + np.sum(z**4, axis=1)
    np.testing.assert_allclose(Cg @ g, a + lap.STEP**2, rtol=0, atol=1e-10)
    second = (Ch @ g).reshape(q, q)
    expected = -A + 2 * lap.STEP**2 * np.eye(q)
    np.testing.assert_allclose(second, expected, rtol=0, atol=1e-8)


# -- a closed form: no marker effect -----------------------------------------


@pytest.mark.parametrize("q", [2, 3, 4])
def test_laplace_is_exact_without_a_marker_effect(q):
    n = 12
    data = grids.simulate(n, seed=51, q=q)
    exact = grids._closed_form_neg_ll(PATHS[q], data)
    model = grids.model_at(PATHS[q], "laplace", effect=0.0)
    # (the curvature is by differences of step 0.01: 1e-8 an item)
    assert grids.neg_ll(model, data) == pytest.approx(exact, abs=1e-8 * n)


def _knots_closed_form(data):
    total = 0.0
    for x, c, readings in zip(data["x"], data["c"], data["markers"]):
        ages = np.array(list(readings), dtype=float)
        y = np.array(list(readings.values()), dtype=float)
        X = knots.KnotPath.basis(ages)
        cov = X @ np.diag(knots.SDS**2) @ X.T
        cov = cov + knots.NOISE**2 * np.eye(ages.size)
        total += multivariate_normal(X @ knots.MEANS, cov).logpdf(y)
        total -= (x / knots.ALPHA) ** knots.BETA
        if c == 0:
            total += np.log(knots.BETA / knots.ALPHA) + (
                knots.BETA - 1
            ) * np.log(x / knots.ALPHA)
    return -total


def test_laplace_is_exact_in_six_dimensions_without_a_marker_effect():
    n = 10
    data = knots.simulate(n, seed=52, effect=0.0)
    exact = _knots_closed_form(data)
    model = knots_at(effect=0.0)
    assert grids.neg_ll(model, data) == pytest.approx(exact, abs=1e-8 * n)
    # ... and so are its predictions: with no marker effect the item's
    # survival is the Weibull's, whatever its readings
    x = np.array([1.5, 2.0, 2.5])
    readings = {0.0: 3.1, 0.5: 3.6, 1.0: 3.8}
    weibull = np.exp(
        -((x / knots.ALPHA) ** knots.BETA) + (1.0 / knots.ALPHA) ** knots.BETA
    )
    np.testing.assert_allclose(
        model.sf(x, markers=readings, given=1.0), weibull, rtol=1e-8
    )
    prediction = model.predict(markers=readings, given=1.0)
    hazard = (knots.BETA / knots.ALPHA) * (x / knots.ALPHA) ** (knots.BETA - 1)
    np.testing.assert_allclose(prediction.hf(x), hazard, rtol=1e-8)


# -- with a marker effect: against the grids ---------------------------------


@pytest.mark.parametrize("q", [2, 3])
@pytest.mark.parametrize(
    "every, noise, bound",
    [(0.1, 0.2, 5e-6), (0.5, 0.6, 2e-3)],
    ids=["dense", "sparse"],
)
def test_laplace_against_seven_and_eleven_points(
    monkeypatch, q, every, noise, bound
):
    # First measured (seed 32, fixed before): 9.6e-7 and 7.2e-7 an item
    # read every 0.1 years with noise 0.2 (up to 31 readings an item),
    # 4.5e-4 and 5.3e-4 read every 0.5 with noise 0.6 (up to 7); the
    # 7-point grid is within 1e-8 an item of the 11-point one on both.
    n = 40
    data = grids.simulate(n, seed=32, q=q, every=every, noise=noise)
    laplace = grids.neg_ll(
        grids.model_at(PATHS[q], "laplace", noise=noise), data
    )
    seven = grids.neg_ll(grids.model_at(PATHS[q], "grid", noise=noise), data)
    monkeypatch.setattr(lik, "GH_FIT", 11)
    eleven = grids.neg_ll(grids.model_at(PATHS[q], "grid", noise=noise), data)
    assert abs(seven - eleven) < 1e-7 * n
    assert laplace != seven
    assert abs(laplace - seven) < bound * n
    assert abs(laplace - eleven) < bound * n


def test_laplace_agrees_with_the_grid_for_the_accelerated_part():
    n = 30
    data = grids.simulate(n, seed=53, q=2)
    values = {}
    for integration in ("grid", "laplace"):
        model = grids.model_at(LINE, integration, effect=0.4)
        model.survival = "AFT"
        values[integration] = grids.neg_ll(model, data)
    assert values["laplace"] != values["grid"]
    assert abs(values["laplace"] - values["grid"]) < 5e-6 * n


def test_laplace_agrees_with_the_grid_for_interval_censored_items():
    # every other failure known only to within half a year
    n = 30
    data = grids.simulate(n, seed=54, q=2)
    x = np.column_stack([data["x"], data["x"]])
    c = data["c"].copy()
    for i in np.flatnonzero(c == 0)[::2]:
        x[i] = [0.5 * np.floor(x[i, 0] / 0.5), 0.5 * np.ceil(x[i, 0] / 0.5)]
        c[i] = 2
        data["markers"][i] = {
            age: y for age, y in data["markers"][i].items() if age <= x[i, 0]
        }
    data = dict(x=x, c=c, markers=data["markers"])
    values = {
        integration: grids.neg_ll(grids.model_at(LINE, integration), data)
        for integration in ("grid", "laplace")
    }
    assert np.isfinite(values["laplace"])
    assert abs(values["laplace"] - values["grid"]) < 5e-5 * n


# -- its derivatives ---------------------------------------------------------


def _differences(f, u, h=1e-4):
    """The gradient of ``f`` at ``u`` by fourth-order central
    differences."""
    out = np.zeros(u.size)
    for j in range(u.size):
        e = np.zeros(u.size)
        e[j] = h
        out[j] = (
            8 * (f(u + e) - f(u - e)) - (f(u + 2 * e) - f(u - 2 * e))
        ) / (12 * h)
    return out


@functools.cache
def _sparse_three():
    data = grids.simulate(40, seed=32, q=3, every=0.5, noise=0.6)
    return grids.model_at(QuadraticPath, "laplace", noise=0.6), data


def test_the_gradient_is_the_approximations_own():
    # The stencils are held while autograd differentiates; the modes
    # they are centred at move with the parameters. The gradient must
    # still be that of the approximation with each mode found again.
    model, data = _sparse_three()
    f, parts, arrays, u = placed_again(model, data)
    placed, _ = lik.place(u, parts, arrays, (None, None))
    value, grad = value_and_grad(
        lambda v: lik.neg_ll(v, parts, arrays, placed)
    )(u)
    assert value == f(u)
    expected = _differences(f, u)
    size = np.max(np.abs(expected))
    # (first measured: 7.7e-7 of the largest entry)
    np.testing.assert_allclose(grad, expected, rtol=0, atol=1e-5 * size)
    # without the implicit-function term it is not (1.8e-2 of it)
    held = copy.copy(placed[0])
    held.c = np.zeros_like(held.c)
    _, short = value_and_grad(
        lambda v: lik.neg_ll(v, parts, arrays, (held, None))
    )(u)
    assert np.max(np.abs(short - expected)) > 1e-3 * size


def test_the_curvature_is_the_approximations_own():
    # the Hessian the standard errors come from, against second
    # differences of the approximation along three directions
    model, data = _sparse_three()
    f, parts, arrays, u = placed_again(model, data)

    def objective(v, placed):
        return lik.neg_ll(v, parts, arrays, placed)

    def evaluated(v, starts):
        placed, starts = lik.place(v, parts, arrays, starts)
        return float(objective(v, placed)), placed, starts

    from autograd import hessian

    placed, _ = lik.place(u, parts, arrays, (None, None))
    H = joint_ph._laplace_curvature(
        u, placed, value_and_grad(objective), hessian(objective), evaluated
    )
    np.testing.assert_allclose(H, H.T, rtol=0, atol=1e-12)
    sd = 1.0 / np.sqrt(np.diag(H))
    rng = np.random.default_rng(55)
    for _ in range(3):
        direction = sd * rng.standard_normal(u.size) / np.sqrt(u.size)
        h = 0.05
        second = (
            f(u + h * direction) - 2 * f(u) + f(u - h * direction)
        ) / h**2
        assert direction @ H @ direction == pytest.approx(second, rel=2e-3)


# -- fits --------------------------------------------------------------------


def _fit(fitter, integration, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitter.fit(*args, integration=integration, **kwargs)


@functools.cache
def _pumps(integration):
    """The recovery study's pumps (read monthly), a fleet of 150."""
    data = _fleet.fleet(n=150, seed=56)
    return _fit(JointPH(Weibull, LinearPath), integration, data)


def _close_fits(laplace, grid):
    assert laplace.integration == "laplace" and grid.integration == "grid"
    assert laplace.maximum == grid.maximum == "verified"
    se = grid.standard_errors()
    shift = (laplace.params - grid.params) / se
    # a tenth of a standard error: what the approximation is held to
    # where the items have five or more readings
    assert np.max(np.abs(shift)) < 0.1, shift
    np.testing.assert_allclose(laplace.standard_errors(), se, rtol=0.03)


def test_a_two_dimension_fit_agrees_with_the_grids():
    laplace, grid = _pumps("laplace"), _pumps("grid")
    _close_fits(laplace, grid)
    assert "Integration         : Laplace's approximation" in repr(laplace)


def test_a_three_dimension_fit_agrees_with_the_grids():
    grid, data = grids._three()
    laplace = _fit(JointPH(Weibull, QuadraticPath), "laplace", **data)
    _close_fits(laplace, grid)
    assert abs(laplace.neg_ll() - grid.neg_ll()) < 1e-5 * len(data["x"])
    # ... saved and restored as a Laplace fit
    saved = laplace.to_dict()
    assert saved["integration"] == "laplace"
    restored = JointModel.from_dict(saved)
    assert restored.integration == "laplace"
    x, readings = [1.5, 2.5], {0.0: 3.1, 1.0: 3.8}
    np.testing.assert_array_equal(
        restored.sf(x, markers=readings, given=1.0),
        laplace.sf(x, markers=readings, given=1.0),
    )


def test_a_laplace_fit_predicts_on_the_grid_up_to_five_dimensions():
    # one item's grid is cheap and more accurate: the same parameters
    # give the same prediction whichever way they were found
    laplace = _pumps("laplace")
    as_grid = copy.deepcopy(laplace)
    as_grid.integration = "grid"
    readings = {0.0: 3.1, 60.0: 3.6, 120.0: 3.8}
    x = [200.0, 400.0, 600.0]
    np.testing.assert_array_equal(
        laplace.sf(x, markers=readings, given=120.0),
        as_grid.sf(x, markers=readings, given=120.0),
    )
    np.testing.assert_array_equal(
        laplace.cb(x, markers=readings, given=120.0),
        as_grid.cb(x, markers=readings, given=120.0),
    )


#: The six-parameter fleet's items; its seed (57) was fixed before a fit.
SIX_N = 120


@functools.cache
def _six(repeat=0):
    data = knots.simulate(SIX_N, seed=57)
    fitter = JointPH(Weibull, KNOT_PATH, random_effects="diagonal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitter.fit(**data, marker_reference=knots.CENTRE)


def _six_truth(model):
    names = list(KNOT_PATH.parameter_names)
    values = dict(zip(names, knots.MEANS))
    values.update({f"sd_{n}": s for n, s in zip(names, knots.SDS)})
    values.update(
        sd_noise=knots.NOISE,
        alpha=knots.ALPHA,
        beta=knots.BETA,
        marker_effect=knots.EFFECT,
    )
    return np.array([values[name] for name in model.parameter_names])


@pytest.mark.calibration
def test_a_six_dimension_fleet_is_recovered():
    model = _six()
    assert model.integration == "laplace"
    assert model.varying == list(KNOT_PATH.parameter_names)
    assert model.maximum == "verified"
    z = (model.params - _six_truth(model)) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))
    assert "Integration         : Laplace's approximation" in repr(model)
    # it predicts, by Laplace's approximation of each integral
    readings = {0.0: 3.1, 0.5: 3.6, 1.0: 3.9}
    x = [1.5, 2.0, 2.5]
    sf = model.sf(x, markers=readings, given=1.0)
    assert np.all(np.diff(sf) < 0) and 0 < sf[-1] < sf[0] < 1
    band = model.cb(x, markers=readings, given=1.0)
    assert np.all(band[:, 0] < sf) and np.all(sf < band[:, 1])
    prediction = model.predict(markers=readings, given=1.0)
    assert np.all(prediction.hf(x) > 0)
    low, high = prediction.marker_interval([0.5, 2.0]).T
    marker = prediction.marker([0.5, 2.0])
    assert np.all(low < marker) and np.all(marker < high)
    # ... kept with pickle (a path of one's own has no name to save by)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.integration == "laplace"
    np.testing.assert_array_equal(
        restored.sf(x, markers=readings, given=1.0), sf
    )


@pytest.mark.calibration
def test_a_six_dimension_fit_is_the_same_twice():
    first, second = _six(0), _six(1)
    assert first is not second
    np.testing.assert_array_equal(first.params, second.params)
    np.testing.assert_array_equal(
        first.standard_errors(), second.standard_errors()
    )
    assert first.neg_ll() == second.neg_ll()


def test_six_dimension_predictions_at_given_parameters():
    # (no fit: the model at the truth) a marker running high lowers the
    # survival, and the smoothed marker follows the readings
    model = knots_at()
    x = [1.5, 2.0, 2.5]
    high = model.sf(x, markers={0.0: 3.6, 0.5: 4.2, 1.0: 4.6}, given=1.0)
    low = model.sf(x, markers={0.0: 2.6, 0.5: 2.9, 1.0: 3.0}, given=1.0)
    assert np.all(high < low)
    assert np.all(np.diff(high) < 0) and 0 < high[-1] < high[0] < 1
    dense = {age: 3.0 + 0.5 * age for age in np.arange(0.0, 1.01, 0.05)}
    prediction = model.predict(markers=dense, given=1.0)
    np.testing.assert_allclose(prediction.marker([0.6]), 3.3, atol=0.1)


# -- the option --------------------------------------------------------------


def test_laplace_is_the_default_beyond_five_varying_parameters():
    assert lik.check_integration(None, 5) == "grid"
    assert lik.check_integration(None, 6) == "laplace"
    assert lik.check_integration(None, 12) == "laplace"
    assert lik.check_integration("laplace", 2) == "laplace"


@pytest.mark.parametrize("factory", [JointPH, JointAFT])
def test_a_grid_beyond_five_varying_parameters_is_refused(factory):
    fitter = factory(Weibull, KNOT_PATH)
    assert len(fitter.varying) == 6
    data = knots.simulate(5, seed=58)
    for grid in ("grid", "pruned"):
        with pytest.raises(
            ValueError, match=r"integration='laplace'.*varying=\(\.\.\.\)"
        ):
            fitter.fit(**data, integration=grid)


def test_laplace_refuses_truncated_items():
    data = _fleet.fleet(n=40, seed=59, entry=True)
    with pytest.raises(
        ValueError, match=r"does not take truncated items \(\d+ here.*'grid'"
    ):
        JointPH(Weibull, LinearPath).fit(data, integration="laplace")


def test_laplace_warns_where_many_items_have_few_readings():
    # pumps read once a year have at most three readings each
    sparse = _fleet.fleet(n=40, seed=60, every=365.0)
    with pytest.warns(
        UserWarning,
        match=r"40 of the 40 items \(100%\) have at most 3 readings.*"
        r"integration='grid'",
    ):
        JointPH(Weibull, LinearPath).fit(sparse, integration="laplace")
    # ... read monthly, few of them have
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        JointPH(Weibull, LinearPath).fit(
            _fleet.fleet(n=40, seed=60), integration="laplace"
        )
    assert not [w for w in caught if "readings" in str(w.message)]


def test_six_parameters_need_items_with_six_readings():
    # as with fewer: the items' own paths are what the fit starts from
    data = knots.simulate(12, seed=61, every=0.75)
    assert max(len(m) for m in data["markers"]) == 5
    with pytest.raises(ValueError, match="readings at 6 or more ages"):
        JointPH(Weibull, KNOT_PATH).fit(**data)


def test_fit_from_df_takes_laplace():
    import pandas as pd

    data = grids.simulate(60, seed=34, q=3)
    df = pd.DataFrame(
        {"x": data["x"], "c": data["c"], "readings": data["markers"]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointPH(Weibull, QuadraticPath).fit_from_df(
            df, "x", "readings", c_col="c", integration="laplace"
        )
    assert model.integration == "laplace"
    direct = _fit(JointPH(Weibull, QuadraticPath), "laplace", **data)
    np.testing.assert_array_equal(model.params, direct.params)
