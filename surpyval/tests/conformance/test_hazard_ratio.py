"""The hazard ratio ``hr``: a model option, beside ``gamma``, ``lfp_p``
and ``f0``.

Every registered parametric distribution (with an offset, a
limited-failure population and zero inflation) takes ``hr`` or refuses
it, saying what to do instead. For each, at ``hr`` in :data:`RATIOS`:

- **identities**: ``Hf`` is ``hr`` times the model's without it, ``sf``
  its survival raised to ``hr``, ``ff`` one less that, and ``hf`` ``hr``
  times the model's hazard (a discrete model's ``1 - (1 - h) ** hr``);
- **splitting**: ``R ** (1 - b) * R ** b = R``, the models at ``hr = 1 -
  b`` and ``hr = b`` together being the model;
- **inversion**: ``isf(sf(t)) == t`` and ``qf(ff(t)) == t``;
- **draws**: a seeded ``random`` is ``qf`` of the same uniforms;
- **unchanged**: ``hr = 1`` gives every function and draw of the model
  without it, to the last bit;
- **saving**: the model round-trips through ``to_dict`` and pickle, its
  dict stamped schema 5 (which SurPyval 0.28 refuses).

The values far into the tails are checked against closed forms in
``univariate/parametric/test_hazard_ratio.py``.
"""

import pickle
import warnings

import numpy as np
import pytest

import surpyval
from surpyval.tests.conformance.registry import CASES, Q_PROBS, fitted

RATIOS = (0.3, 4.0)
RTOL = 1e-13

_PROBABILITY = "a probability of an event, not a life with a hazard"
#: The cases whose distribution refuses ``hr``, and why.
REFUSED = {
    "Binomial": _PROBABILITY,
    "Bernoulli": _PROBABILITY,
    "FixedEventProbability": _PROBABILITY,
    "ExactEventTime": _PROBABILITY,
}

_OURS = [c for c in CASES if c.model_class == "surpyval.Parametric"]
TAKEN = [pytest.param(c, id=c.name) for c in _OURS if c.name not in REFUSED]
REFUSING = [pytest.param(c, id=c.name) for c in _OURS if c.name in REFUSED]


def test_refused_cases_are_registered():
    names = {c.name for c in _OURS}
    assert set(REFUSED) <= names, set(REFUSED) - names


def _query(case):
    x = np.asarray(case.x, dtype=float).ravel()
    finite = x[np.isfinite(x)]
    grid = np.linspace(finite.min(), finite.max(), 20)
    out = np.concatenate([x, grid])
    model = fitted(case)
    if model.dist.discrete:
        out = np.unique(np.floor(out))
    return out


def _with(case, hr):
    model = fitted(case)
    return model, model.with_params(model.params, hr=hr)


@pytest.mark.parametrize("hr", RATIOS)
@pytest.mark.parametrize("case", TAKEN)
def test_functions_are_the_models_scaled(case, hr):
    base, model = _with(case, hr)
    t = _query(case)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        H0 = np.asarray(base.Hf(t), dtype=float)
        np.testing.assert_allclose(model.Hf(t), hr * H0, rtol=RTOL, atol=0)
        np.testing.assert_allclose(
            model.sf(t), np.exp(-hr * H0), rtol=RTOL, atol=0
        )
        np.testing.assert_allclose(
            model.ff(t), -np.expm1(-hr * H0), rtol=RTOL, atol=0
        )
        h0 = np.asarray(base.hf(t), dtype=float)
        if base.dist.discrete:
            want = -np.expm1(hr * np.log1p(-h0))
        else:
            want = hr * h0
            if base.zi:
                want = np.where(t == 0, 1 - (1 - base.f0) ** hr, want)
        np.testing.assert_allclose(model.hf(t), want, rtol=1e-12, atol=0)
        # the density integrates the distribution function
        if not base.dist.discrete:
            inside = (model.sf(t) > 1e-12) & (t > 0) & (model.ff(t) > 1e-12)
            if base.zi:
                inside &= t > 0
            want = hr * h0 * np.asarray(model.sf(t))
            np.testing.assert_allclose(
                np.asarray(model.df(t))[inside],
                want[inside],
                rtol=1e-12,
                atol=0,
            )


@pytest.mark.parametrize("case", TAKEN)
def test_split_hazard_ratios_make_the_model(case):
    base = fitted(case)
    t = _query(case)
    for b in (0.1, 0.5, 0.9):
        lo = base.with_params(base.params, hr=1 - b)
        hi = base.with_params(base.params, hr=b)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            np.testing.assert_allclose(
                np.asarray(lo.Hf(t)) + np.asarray(hi.Hf(t)),
                base.Hf(t),
                rtol=RTOL,
                atol=0,
            )
            np.testing.assert_allclose(
                np.asarray(lo.sf(t)) * np.asarray(hi.sf(t)),
                base.sf(t),
                rtol=RTOL,
                atol=0,
            )


@pytest.mark.parametrize("hr", RATIOS)
@pytest.mark.parametrize("case", TAKEN)
def test_isf_and_qf_invert(case, hr):
    _, model = _with(case, hr)
    t = _query(case)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        s = np.asarray(model.sf(t), dtype=float)
        f = np.asarray(model.ff(t), dtype=float)
        if model.dist.discrete:
            # the first point with sf(k) <= s
            ok = (s > 0) & (s < 1)
            np.testing.assert_array_equal(model.isf(s[ok]), t[ok])
            return
        # where the time is pinned: sf moves with it, and inside the
        # support (an offset's, a zero inflation's mass at 0)
        h = np.asarray(model.hf(t), dtype=float)
        lower = model.support[0] + (model.gamma if model.offset else 0)
        firm = (t > lower) & (h > 0) & np.isfinite(h)
        up = firm & (s > 1e-300) & (s < 0.5)
        np.testing.assert_allclose(model.isf(s[up]), t[up], rtol=1e-10)
        low = firm & (f > 1e-300) & (f < 0.5)
        np.testing.assert_allclose(model.qf(f[low]), t[low], rtol=1e-10)
        # qf is isf of 1 - p where 1 - p is exact
        np.testing.assert_allclose(
            model.qf(Q_PROBS), model.isf(1 - np.asarray(Q_PROBS)), rtol=RTOL
        )


@pytest.mark.parametrize("case", TAKEN)
def test_draws_are_qf_of_the_uniforms(case):
    _, model = _with(case, 0.3)
    np.random.seed(11)
    drawn = model.random(40)
    np.random.seed(11)
    with np.errstate(all="ignore"):
        want = model.qf(np.random.random_sample(40))
    np.testing.assert_array_equal(drawn, want)
    np.testing.assert_array_equal(
        model.random(25, random_state=5), model.random(25, random_state=5)
    )


@pytest.mark.parametrize("case", TAKEN)
def test_a_ratio_of_one_changes_nothing(case):
    base = fitted(case)
    one = base.with_params(base.params, hr=1)
    same = base.with_params(base.params)
    assert one.hr == 1 and "hr" not in one.extras
    t = _query(case)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for fname in ("sf", "ff", "df", "hf", "Hf"):
            np.testing.assert_array_equal(
                getattr(one, fname)(t), getattr(same, fname)(t), fname
            )
        np.testing.assert_array_equal(one.qf(Q_PROBS), same.qf(Q_PROBS))
        np.testing.assert_array_equal(one.isf(Q_PROBS), same.isf(Q_PROBS))
        np.random.seed(2)
        a = one.random(30)
        np.random.seed(2)
        np.testing.assert_array_equal(a, same.random(30))
    assert one.to_dict() == same.to_dict()


@pytest.mark.parametrize("case", TAKEN)
def test_saved_with_its_ratio(case):
    _, model = _with(case, 0.3)
    d = model.to_dict()
    assert d["hr"] == 0.3 and d["schema"] == 5
    back = surpyval.from_dict(d)
    again = pickle.loads(pickle.dumps(model))
    t = _query(case)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for other in (back, again):
            assert other.hr == 0.3
            np.testing.assert_array_equal(other.sf(t), model.sf(t))
            np.testing.assert_array_equal(other.extras, model.extras)
    assert "Hazard ratio (hr)" in repr(model)


@pytest.mark.parametrize("case", REFUSING)
def test_refusing_cases_say_so(case):
    model = fitted(case)
    with pytest.raises(ValueError, match="'hr'"):
        model.with_params(model.params, hr=2.0)
