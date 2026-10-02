"""
One name per option in regression and competing risks (#422, principle
21), and a seed for every regression ``random`` (#389).

The old names (CoxPH's ``method``, ``id_col``, ``time_col``,
BuckleyJames's ``seed``, the competing-risks ``how`` and ``cause``, and
the fitted ``CompetingRisksProportionalHazards.how`` and
``CompetingRisks.method``) were removed in v0.22.0; see
test_removed_arguments.py. The tests here check that the new names take
effect, that positional calls read the same arguments, and that the saved
files keep the old keys so older files still load.
"""

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import AcceleratedLife, Weibull
from surpyval.life_models import Power
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    CompetingRisksProportionalHazards,
    FineGray,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _tied():
    # Heavy ties, where the Breslow and Efron fits differ.
    x = np.array([1, 1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6], float)
    Z = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0], float)[:, None]
    c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    return x, Z, c


def _tvc():
    """The same subjects as start-stop rows and as a covariate timeline:
    a stress that switches on part way through follow-up. The times are
    whole numbers, so there are ties for the tie methods to differ on."""
    rng = np.random.default_rng(3)
    stop_rows, timeline = [], []
    for i in range(40):
        switch = float(rng.integers(1, 6))
        t0 = np.ceil(rng.weibull(2.0) * 8.0)
        if t0 <= switch:
            stop_rows.append((i, 0.0, t0, 0, 0.0))
            timeline += [(i, 0.0, 0.0, 1), (i, t0, np.nan, 0)]
        else:
            # A stressed unit ages twice as fast after the switch.
            t = switch + np.ceil((t0 - switch) / 2.0)
            stop_rows += [(i, 0.0, switch, 1, 0.0), (i, switch, t, 0, 1.0)]
            timeline += [(i, 0.0, 0.0, 1), (i, switch, 1.0, 1)]
            timeline.append((i, t, np.nan, 0))
    ss = pd.DataFrame(stop_rows, columns=["id", "xl", "xr", "c", "z"])
    tl = pd.DataFrame(timeline, columns=["id", "time", "z", "c"])
    return ss, tl


def _cr():
    rng = np.random.default_rng(0)
    Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
    t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * Z[:, 0])))
    t_b = rng.exponential(1 / 0.05, 200)
    t_c = rng.uniform(0, 20, 200)
    x = np.minimum(np.minimum(t_a, t_b), t_c).round(3)
    first = np.where(t_a < t_b, "a", "b")
    e = np.where(t_c < np.minimum(t_a, t_b), None, first)
    return x, Z, e


# ---------------------------------------------------------------------------
# The renamed options
# ---------------------------------------------------------------------------
def _cox(**kw):
    x, Z, c = _tied()
    return sp.CoxPH.fit(x, Z, c=c, **kw).beta


def _cox_tvc(**kw):
    ss, _ = _tvc()
    return sp.CoxPH.fit_tvc(
        ss["id"], ss["xl"], ss["xr"], ss["c"], ss[["z"]], **kw
    ).beta


def _crph(**kw):
    x, Z, e = _cr()
    return CompetingRisksProportionalHazards.fit(x, Z, e, **kw).betas


def _fine_gray(**kw):
    x, Z, e = _cr()
    return FineGray.fit(x, Z, e, **kw).beta


def _gray(**kw):
    x, Z, e = _cr()
    return sp.gray_test(x, e, Z[:, 0], **kw).statistic


def test_renamed_options_take_effect():
    # Values other than the defaults change the answer, so a renamed
    # option that were dropped would show.
    assert not np.allclose(_cox(tie_method="breslow"), _cox())
    assert not np.allclose(
        _cox_tvc(tie_method="breslow"), _cox_tvc(), atol=1e-4
    )
    assert not np.allclose(
        _crph(model="Fine-Gray"), _crph(model="Cox"), atol=1e-3
    )
    assert not np.allclose(_fine_gray(event="b"), _fine_gray(event="a"))


def test_positional_calls_are_unchanged():
    # Only the names changed: a positional call reads the same argument.
    x, Z, c = _tied()
    np.testing.assert_array_equal(
        sp.CoxPH.fit(x, Z, c, None, None, "breslow").beta,
        _cox(tie_method="breslow"),
    )
    xc, Zc, e = _cr()
    np.testing.assert_array_equal(
        FineGray.fit(xc, Zc, e, None, None, "b").beta, _fine_gray(event="b")
    )
    assert sp.gray_test(xc, e, Zc[:, 0], "b").statistic == _gray(event="b")


def test_crph_model_attribute():
    x, Z, e = _cr()
    for kind in ("Cox", "Fine-Gray"):
        fitted = CompetingRisksProportionalHazards.fit(x, Z, e, model=kind)
        assert fitted.model == kind
        # The old alias ``.how`` is gone ...
        assert not hasattr(fitted, "how")
        # ... but the model is saved under the old key, so files from
        # before the rename load.
        assert fitted.to_dict()["how"] == kind
        assert sp.from_dict(fitted.to_dict()).model == kind


def test_crph_rejects_an_unknown_model_by_its_name():
    x, Z, e = _cr()
    with pytest.raises(ValueError, match="'model' must be one of"):
        CompetingRisksProportionalHazards.fit(x, Z, e, model="Weibull")


# ---------------------------------------------------------------------------
# random_state for the regression random (#389)
# ---------------------------------------------------------------------------
def _ph():
    rng = np.random.default_rng(1)
    Z = rng.binomial(1, 0.5, 80)[:, None].astype(float)
    x = rng.weibull(2, 80) * 10 * np.exp(-0.5 * Z[:, 0])
    return sp.WeibullPH.fit(x, Z)


def _ah():
    rng = np.random.default_rng(1)
    Z = rng.binomial(1, 0.5, 80)[:, None].astype(float)
    x = rng.weibull(2, 80) * 10
    return sp.WeibullAH.fit(x, Z)


def _al():
    rng = np.random.default_rng(1)
    stress = np.repeat([20.0, 30.0, 40.0], 30)[:, None]
    x = rng.weibull(3, 90) * 10 * (100.0 / stress[:, 0])
    return AcceleratedLife(Weibull, Power).fit(x, Z=stress)


MODELS = {"PH": _ph, "AH": _ah, "AL": _al}
# Two covariate rows (two stresses for the accelerated life model).
ROWS = {"PH": [[0.0], [1.0]], "AH": [[0.0], [1.0]], "AL": [20.0, 40.0]}


@pytest.fixture(autouse=True)
def _restore_global_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.mark.parametrize("kind", MODELS)
def test_random_seeded_by_random_state(kind):
    model, Z = MODELS[kind](), ROWS[kind]
    np.random.seed(1)
    a = model.random(6, Z, random_state=7)[0]
    after = np.random.uniform(size=3)
    np.random.seed(1)
    untouched = np.random.uniform(size=3)
    np.random.seed(2)
    b = model.random(6, Z, random_state=7)[0]
    # The same seed gives the same draw, whatever the global state, and
    # leaves the global stream alone.
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(after, untouched)
    # An int seed means numpy.random.default_rng(seed).
    np.testing.assert_array_equal(
        model.random(6, Z, random_state=np.random.default_rng(7))[0], a
    )
    # One stream for all the rows: the rows' draws are not the same
    # uniforms over again.
    u_first = model.sf(a[:6], np.tile(np.atleast_1d(Z[0]), (6, 1)))
    u_second = model.sf(a[6:], np.tile(np.atleast_1d(Z[1]), (6, 1)))
    assert not np.allclose(u_first, u_second)
    assert not np.array_equal(a, model.random(6, Z, random_state=8)[0])


@pytest.mark.parametrize("kind", MODELS)
def test_random_without_a_seed_follows_the_global_rng(kind):
    model, Z = MODELS[kind](), ROWS[kind]
    np.random.seed(0)
    a = model.random(6, Z)[0]
    np.random.seed(0)
    b = model.random(6, Z, random_state=None)[0]
    np.random.seed(1)
    c = model.random(6, Z)[0]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_without_a_seed_draws_as_before():
    # random_state=None draws numpy's global uniforms, as random always
    # did, so seeded code gives the same numbers as before the argument.
    model = _ph()
    np.random.seed(3)
    got = model.random(5, [[1.0]])[0]
    np.random.seed(3)
    u = np.random.uniform(0, 1, 5)
    np.testing.assert_allclose(model.sf(got, np.ones((5, 1))), u)


def test_fitter_random_takes_random_state():
    model = _ph()
    a = sp.WeibullPH.random(4, [[1.0]], *model.params, random_state=5)
    b = model.random(4, [[1.0]], random_state=5)
    np.testing.assert_array_equal(a[0], b[0])


def test_competing_risks_how_attribute():
    x, _, e = _cr()
    model = CompetingRisks.fit(x, e, how="Kaplan-Meier")
    assert model.how == "Kaplan-Meier"
    # The old alias ``.method`` is gone; the estimator is still saved
    # under its old key, so older files still load.
    assert not hasattr(model, "method")
    assert model.to_dict()["method"] == "Kaplan-Meier"
    assert CompetingRisks.from_dict(model.to_dict()).how == "Kaplan-Meier"
