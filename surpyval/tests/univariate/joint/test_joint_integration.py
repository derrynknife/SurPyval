"""How a joint model's integral over each item's path parameters is done:
the full grid and the pruned one (``integration=``), and four and five
varying parameters.

The paths are polynomials defined here (a cubic has four parameters, a
quartic five; the built-in paths have at most three). The checks:

- the pruning rule itself: which nodes go, that they hold less than 1e-9
  of the weight, and that the full grid is the one it always was;
- a closed form: with no marker effect and a path linear in its
  parameters the integral is a multivariate normal density of the
  readings times the Weibull's own likelihood, which both grids must
  reproduce in four and five dimensions;
- with a marker effect, the pruned grid against the full one in three,
  four and five dimensions (to 1e-6 an item), and the 7-point grid
  against an 11-point one in four (to the 1e-5 an item of the dense-grid
  tests);
- fits: a three-dimension fit on each grid, and a seeded four-dimension
  fleet recovered on the pruned grid and compared with the full one;
- the option's refusals, the model's record of it, saving and restoring.
"""

import copy
import functools
import itertools
import warnings

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from surpyval import JointAFT, JointModel, JointPH, Weibull
from surpyval.degradation import PathModel, QuadraticPath
from surpyval.univariate.joint import _likelihood as lik
from surpyval.univariate.joint.joint_model import parameter_names
from surpyval.univariate.joint.joint_ph import search_space


class PolynomialPath(PathModel):
    """``y = a + b t + c t^2 + ...``: a path of one's own, linear in its
    parameters (module level, to pickle)."""

    linear_in_parameters = True

    def __init__(self, degree):
        self.name = {3: "Cubic", 4: "Quartic", 5: "Quintic"}[degree]
        self.parameter_names = list("abcdef")[: degree + 1]

    def path(self, x, *params):
        x = np.asarray(x, dtype=float)
        out = 0.0
        for k, value in enumerate(params):
            out = out + value * x**k
        return out

    def inv_path(self, y, *params):
        return np.full(np.shape(y), np.nan)

    def jacobian(self, x, *params):
        x = np.asarray(x, dtype=float)
        return np.column_stack([x**k for k in range(len(params))])

    def _initial_guess(self, x, y):
        return [float(np.mean(y))] + [0.0] * (len(self.parameter_names) - 1)


CUBIC, QUARTIC = PolynomialPath(3), PolynomialPath(4)

#: The truth of the simulated fleets, ages in years: each item's
#: polynomial coefficients are independent normals about ``MEANS`` (the
#: first ``q`` of them), read with noise; the hazard is Weibull
#: proportional hazards in the path's value.
MEANS = np.array([3.0, 1.0, -0.6, 0.25, -0.03])
SDS = np.array([0.5, 0.4, 0.3, 0.1, 0.03])
NOISE, ALPHA, BETA, EFFECT, CENTRE = 0.2, 2.5, 2.0, 0.8, 3.6


def simulate(n, seed, q, follow=3.0, every=0.1, noise=NOISE):
    """``n`` items of a polynomial with ``q`` coefficients, read every
    ``every`` years (with noise of sd ``noise``) until they fail or reach
    ``follow``: the arrays of a fit (``x``, ``c``, ``markers``)."""
    rng = np.random.default_rng(seed)
    theta = MEANS[:q] + SDS[:q] * rng.standard_normal((n, q))
    s = np.linspace(0, follow, 3001)
    m = theta @ np.stack([s**k for k in range(q)])
    h = (BETA / ALPHA) * (s / ALPHA) ** (BETA - 1)
    h = h * np.exp(EFFECT * (m - CENTRE))
    steps = (h[:, 1:] + h[:, :-1]) / 2 * np.diff(s)
    H = np.concatenate([np.zeros((n, 1)), np.cumsum(steps, axis=1)], axis=1)
    target = rng.exponential(size=n)
    x, c, markers = [], [], []
    for i in range(n):
        failed = H[i, -1] > target[i]
        life = float(np.interp(target[i], H[i], s)) if failed else follow
        ages = np.arange(0.0, life + 1e-9, every)
        y = PolynomialPath.path(None, ages, *theta[i]) + rng.normal(
            0, noise, ages.size
        )
        x.append(life)
        c.append(0 if failed else 1)
        markers.append(dict(zip(ages.tolist(), y.tolist())))
    return dict(x=np.array(x), c=np.array(c), markers=markers)


def model_at(path, integration, effect=EFFECT, corr=0.0, noise=NOISE):
    """A joint model of ``path`` (every parameter varying) at the truth,
    every correlation ``corr``, without a fit."""
    names = list(path.parameter_names)
    q = len(names)
    values = dict(zip(names, MEANS[:q]))
    values.update({f"sd_{n}": s for n, s in zip(names, SDS[:q])})
    values.update(sd_noise=noise, alpha=ALPHA, beta=BETA, marker_effect=effect)
    out = JointModel()
    out.dist = Weibull
    out.path_model = path
    out.varying = names
    out.parameter_names = parameter_names(names, names, [])
    out.params = np.array(
        [values.get(name, corr) for name in out.parameter_names]
    )
    out.feature_names = []
    out.aliased = np.array([], dtype=int)
    out.marker_reference = CENTRE
    out._centre = CENTRE
    out._zbar = np.zeros(0)
    out.integration = integration
    out.k = len(out.params)
    return out


def neg_ll(model, data):
    f, u, _ = search_space(model, **data)
    return float(f(u))


# -- the pruning rule --------------------------------------------------------


def _product_grid(dim, points):
    x, w = np.polynomial.hermite.hermgauss(points)
    z = np.array(list(itertools.product(x, repeat=dim)))
    logw = np.array([sum(t) for t in itertools.product(np.log(w), repeat=dim)])
    return z, logw


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_the_full_grid_is_the_product_grid(dim):
    z, logw = lik.gh_grid(dim, 7)
    ref_z, ref_logw = _product_grid(dim, 7)
    np.testing.assert_array_equal(z, ref_z)
    np.testing.assert_allclose(logw, ref_logw, rtol=0, atol=1e-13)
    assert z.shape == (7**dim, dim)


@pytest.mark.parametrize(
    "dim, points, kept",
    [
        (1, 7, 7),
        (2, 7, 49),
        (3, 7, 343),
        (4, 7, 2321),
        (5, 7, 15415),
        (2, 15, 177),
        (3, 9, 673),
    ],
)
def test_the_pruned_grid_leaves_out_less_than_1e9_of_the_weight(
    dim, points, kept
):
    full_z, full_logw = lik.gh_grid(dim, points)
    z, logw = lik.gh_grid(dim, points, prune=True)
    assert z.shape == (kept, dim) and logw.shape == (kept,)
    total = np.sum(np.exp(full_logw))
    left_out = 1.0 - np.sum(np.exp(logw)) / total
    assert 0.0 <= left_out < 1e-9
    # every node kept is a node of the full grid with its own weight, and
    # each is at least as heavy as any left out
    where = {tuple(row): w for row, w in zip(full_z, full_logw)}
    assert all(where[tuple(row)] == w for row, w in zip(z, logw))
    gone = sorted(set(full_logw) - set(logw))
    if gone:
        assert gone[-1] < logw.min() - 1e-6
        # the lightest nodes kept would have been too much to leave out
        lightest = np.exp(full_logw[full_logw <= logw.min() + 1e-9])
        assert np.sum(lightest) / total >= 1e-9


@pytest.mark.parametrize("dim", [3, 4])
def test_the_pruned_grid_keeps_its_symmetries(dim):
    z, _ = lik.gh_grid(dim, 7, prune=True)
    nodes = {tuple(np.round(row, 12)) for row in z}
    for row in z[:: max(1, len(z) // 200)]:
        for perm in itertools.permutations(range(dim)):
            assert tuple(np.round(row[list(perm)], 12)) in nodes
        assert tuple(np.round(-row, 12)) in nodes
        flipped = row.copy()
        flipped[0] = -flipped[0]
        assert tuple(np.round(flipped, 12)) in nodes


def test_the_default_integration_by_the_number_of_varying_parameters():
    # The full grid, the more accurate, is the default at every size;
    # pruning is only ever chosen.
    assert [lik.check_integration(None, q) for q in range(1, 6)] == [
        "grid"
    ] * 5
    assert lik.check_integration("grid", 5) == "grid"
    assert lik.check_integration("pruned", 2) == "pruned"


# -- a closed form: no marker effect -----------------------------------------


def _closed_form_neg_ll(path, data):
    """With no marker effect and a path linear in its parameters: the
    readings are multivariate normal, ``N(X mu, X D X' + sd_noise^2 I)``,
    and the lifetimes the Weibull's own."""
    q = len(path.parameter_names)
    total = 0.0
    for x, c, readings in zip(data["x"], data["c"], data["markers"]):
        ages = np.array(list(readings), dtype=float)
        y = np.array(list(readings.values()), dtype=float)
        X = path.jacobian(ages, *MEANS[:q])
        cov = X @ np.diag(SDS[:q] ** 2) @ X.T + NOISE**2 * np.eye(ages.size)
        total += multivariate_normal(X @ MEANS[:q], cov).logpdf(y)
        total -= (x / ALPHA) ** BETA
        if c == 0:
            total += np.log(BETA / ALPHA) + (BETA - 1) * np.log(x / ALPHA)
    return -total


@pytest.mark.parametrize(
    "path, q", [(CUBIC, 4), (QUARTIC, 5)], ids=["four", "five"]
)
def test_both_grids_reproduce_the_closed_form_without_a_marker_effect(path, q):
    n = 12 if q == 4 else 8
    data = simulate(n, seed=31, q=q)
    exact = _closed_form_neg_ll(path, data)
    full = neg_ll(model_at(path, "grid", effect=0.0), data)
    pruned = neg_ll(model_at(path, "pruned", effect=0.0), data)
    # the integrand is exactly normal, so the full grid is exact and the
    # pruned one short by the weight it leaves out (below 1e-9 an item)
    assert full == pytest.approx(exact, abs=1e-9 * n)
    assert pruned == pytest.approx(exact, abs=2e-9 * n)
    assert 0.0 <= pruned - full < 1e-9 * n


# -- with a marker effect: the pruned grid against the full one --------------


@pytest.mark.parametrize(
    "path, q, n",
    [(QuadraticPath, 3, 40), (CUBIC, 4, 40), (QUARTIC, 5, 24)],
    ids=["three", "four", "five"],
)
@pytest.mark.parametrize("corr", [0.0, 0.3])
@pytest.mark.parametrize(
    "every, noise", [(0.1, 0.2), (0.5, 0.6)], ids=["dense", "sparse"]
)
def test_the_pruned_likelihood_agrees_with_the_full_grids(
    path, q, n, corr, every, noise
):
    # read often with little noise the readings fix each item's path and
    # the integrand is all but normal; read seldom with much, the
    # lifetime shapes it too
    data = simulate(n, seed=32, q=q, every=every, noise=noise)
    full = neg_ll(model_at(path, "grid", corr=corr, noise=noise), data)
    pruned = neg_ll(model_at(path, "pruned", corr=corr, noise=noise), data)
    assert np.isfinite(full)
    assert abs(pruned - full) < 1e-6 * n
    if q == 3:  # nothing is left out of 7 points in three dimensions
        assert pruned == full


def test_seven_points_agree_with_eleven_in_four_dimensions(monkeypatch):
    # the grid of the fit against a finer one of the same kind (14,641
    # nodes an item): the error of 7 points a dimension itself
    n = 12
    data = simulate(n, seed=33, q=4)
    seven = neg_ll(model_at(CUBIC, "grid"), data)
    pruned = neg_ll(model_at(CUBIC, "pruned"), data)
    monkeypatch.setattr(lik, "GH_FIT", 11)
    eleven = neg_ll(model_at(CUBIC, "grid"), data)
    assert seven != eleven
    assert abs(seven - eleven) < 1e-5 * n
    assert abs(pruned - eleven) < 1e-5 * n


# -- fits ------------------------------------------------------------------


@functools.cache
def _three(integration=None):
    data = simulate(60, seed=34, q=3)
    fitter = JointPH(Weibull, QuadraticPath)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitter.fit(**data, integration=integration), data


def test_three_dimensions_default_to_the_full_grid_and_pruned_agrees():
    default, data = _three()
    assert default.integration == "grid"
    pruned, _ = _three("pruned")
    assert pruned.integration == "pruned"
    assert pruned.maximum == default.maximum
    n = len(data["x"])
    assert abs(pruned.neg_ll() - default.neg_ll()) < 1e-6 * n
    shift = (pruned.params - default.params) / default.standard_errors()
    assert np.max(np.abs(shift)) < 0.05, shift
    # ... and the predictions, each on its own grid
    readings = {0.0: 3.1, 0.5: 3.6, 1.0: 3.8}
    x = [1.5, 2.0, 2.5]
    np.testing.assert_allclose(
        pruned.sf(x, markers=readings, given=1.0),
        default.sf(x, markers=readings, given=1.0),
        rtol=1e-5,
    )


def test_asking_for_the_grid_is_the_default_fit_in_three_dimensions():
    default, data = _three()
    asked = JointPH(Weibull, QuadraticPath).fit(**data, integration="grid")
    np.testing.assert_array_equal(asked.params, default.params)
    assert asked.neg_ll() == default.neg_ll()


@functools.cache
def _four(integration=None):
    data = simulate(FOUR_N, seed=35, q=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return JointPH(Weibull, CUBIC).fit(**data, integration=integration)


#: The items of the four-dimension fleet.
FOUR_N = 60


def _four_truth(model):
    names = list(CUBIC.parameter_names)
    values = dict(zip(names, MEANS[:4]))
    values.update({f"sd_{n}": s for n, s in zip(names, SDS[:4])})
    values.update(sd_noise=NOISE, beta=BETA, marker_effect=EFFECT)
    values["alpha"] = ALPHA * np.exp(
        -EFFECT * (model.marker_reference - CENTRE) / BETA
    )
    return np.array([values.get(name, 0.0) for name in model.parameter_names])


@pytest.mark.calibration
def test_a_four_dimension_fleet_is_recovered_on_the_pruned_grid():
    model = _four("pruned")
    assert model.integration == "pruned"
    assert model.varying == ["a", "b", "c", "d"]
    assert model.maximum == "verified"
    z = (model.params - _four_truth(model)) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))
    assert "Integration         : adaptive Gauss-Hermite, pruned grid" in repr(
        model
    )
    # it predicts (on a pruned grid of its own)
    sf = model.sf([1.5, 2.0, 2.5], markers={0.0: 3.1, 1.0: 3.9}, given=1.0)
    assert np.all(np.diff(sf) < 0) and 0 < sf[-1] < sf[0] < 1


@pytest.mark.calibration
def test_the_four_dimension_fit_agrees_with_the_full_grids():
    pruned, full = _four("pruned"), _four()
    assert full.integration == "grid"
    assert abs(pruned.neg_ll() - full.neg_ll()) < 1e-6 * FOUR_N
    shift = (pruned.params - full.params) / full.standard_errors()
    assert np.max(np.abs(shift)) < 0.05, shift


# -- the option --------------------------------------------------------------


def test_an_unknown_integration_is_refused():
    data = simulate(5, seed=36, q=3)
    fitter = JointPH(Weibull, QuadraticPath)
    for bad in ("laplace", "Grid", 7):
        with pytest.raises(ValueError, match="'grid'.*or 'pruned'"):
            fitter.fit(**data, integration=bad)


@pytest.mark.parametrize("factory", [JointPH, JointAFT])
def test_up_to_five_parameters_may_vary(factory):
    assert factory(Weibull, QUARTIC).varying == ("a", "b", "c", "d", "e")
    quintic = PolynomialPath(5)
    with pytest.raises(ValueError, match=r"at most 5 .* varying=\(\.\.\.\)"):
        factory(Weibull, quintic)
    with pytest.raises(ValueError, match="At most 5 .* Laplace .* got 6"):
        factory(Weibull, quintic, varying=tuple("abcdef"))
    assert len(factory(Weibull, quintic, varying=tuple("abcde")).varying) == 5


def test_the_integration_is_printed_saved_and_restored():
    default, _ = _three()
    pruned, _ = _three("pruned")
    assert "Integration         : adaptive Gauss-Hermite, full grid" in repr(
        default
    )
    assert "pruned grid" in repr(pruned)
    assert default.to_dict()["integration"] == "grid"
    saved = pruned.to_dict()
    assert saved["integration"] == "pruned"
    restored = JointModel.from_dict(saved)
    assert restored.integration == "pruned"
    x, readings = [1.5, 2.5], {0.0: 3.1, 1.0: 3.8}
    np.testing.assert_array_equal(
        restored.sf(x, markers=readings, given=1.0),
        pruned.sf(x, markers=readings, given=1.0),
    )
    # a model saved before the choice existed used the full grid
    del saved["integration"]
    assert JointModel.from_dict(saved).integration == "grid"
    saved["integration"] = "laplace"
    with pytest.raises(ValueError, match="'grid' or 'pruned'"):
        JointModel.from_dict(saved)
    assert copy.deepcopy(pruned).integration == "pruned"


def test_fit_from_df_takes_the_integration():
    import pandas as pd

    data = simulate(60, seed=34, q=3)
    df = pd.DataFrame(
        {"x": data["x"], "c": data["c"], "readings": data["markers"]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointPH(Weibull, QuadraticPath).fit_from_df(
            df, "x", "readings", c_col="c", integration="pruned"
        )
    assert model.integration == "pruned"
    np.testing.assert_array_equal(model.params, _three("pruned")[0].params)
