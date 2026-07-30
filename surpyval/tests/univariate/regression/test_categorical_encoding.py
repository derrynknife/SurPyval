"""
Reference-level (reduced-rank) coding for categoricals in formula fits
(#252). A full one-hot encoding is exactly collinear with the baseline
distribution's scale (or the Cox baseline), leaving the coefficients
non-identified: the likelihood is flat along "shift both dummies,
rescale the baseline". Formulas are now materialised with their
implicit intercept (giving treatment coding) and the intercept column
is dropped.
"""

import json

import numpy as np
import pandas as pd
import pytest

import surpyval as surv
from surpyval import WeibullPH
from surpyval.univariate.regression import BuckleyJames, CoxPH


def _df(seed=5, n=300):
    rng = np.random.default_rng(seed)
    sex = rng.choice(["M", "F"], n)
    age = rng.normal(50, 10, n)
    lam = np.exp(0.5 * (sex == "M") + 0.02 * (age - 50))
    x = 10.0 * (-np.log(rng.uniform(size=n))) ** (1 / 2.0) * lam ** (-1 / 2.0)
    return pd.DataFrame(
        {"time": x, "sex": sex, "age": age, "cens": np.zeros(n)}
    )


def test_categorical_gets_reference_level_coding():
    df = _df()
    m = WeibullPH.fit_from_df(
        df, x_col="time", formula="age + sex", c_col="cens"
    )
    # One column for a two-level categorical, not a full one-hot.
    assert len(m.feature_names) == 2
    assert sum("sex" in f for f in m.feature_names) == 1


def test_categorical_coefficient_is_identified():
    # Before #252 the likelihood was exactly flat along the one-hot ridge:
    # coefficients were optimizer-path noise and CIs were huge/NaN. The
    # reference-coded coefficient is the log-HR of M vs F (true 0.5).
    df = _df()
    m = WeibullPH.fit_from_df(
        df, x_col="time", formula="age + sex", c_col="cens"
    )
    cb = m.param_cb("beta_1")
    # Identifiability is the point: a finite, *narrow* interval (the
    # one-hot ridge produced huge or NaN intervals) around a positive
    # log-hazard-ratio for males.
    assert np.all(np.isfinite(cb))
    assert (cb[1] - cb[0]) < 1.5
    beta_sex = float(m.params[-1])
    assert 0.0 < beta_sex < 1.5
    assert cb[0] < beta_sex < cb[1]


def test_serialisation_round_trip_with_categorical():
    df = _df()
    new = pd.DataFrame({"age": [60.0, 40.0], "sex": ["M", "F"]})
    for fitter in (WeibullPH, CoxPH):
        m = fitter.fit_from_df(
            df, x_col="time", formula="age + sex", c_col="cens"
        )
        restored = surv.from_dict(json.loads(json.dumps(m.to_dict())))
        assert np.allclose(m.sf([5, 10], new), restored.sf([5, 10], new))


def test_interaction_terms_are_reduced_rank():
    df = _df()
    m = WeibullPH.fit_from_df(
        df, x_col="time", formula="age * sex", c_col="cens"
    )
    # age, sex[T.M], age:sex[T.M] -- no duplicated one-hot pairs.
    assert len(m.feature_names) == 3


def test_explicit_no_intercept_opts_out():
    # "0 + ..." keeps the old full-rank behaviour for users who want
    # cell-means coding on their own responsibility.
    df = _df()
    m = WeibullPH.fit_from_df(
        df, x_col="time", formula="0 + sex", c_col="cens"
    )
    assert len(m.feature_names) == 2
    assert all("sex[" in f for f in m.feature_names)


def test_numeric_only_formula_unchanged():
    df = _df()
    m = WeibullPH.fit_from_df(df, x_col="time", formula="age", c_col="cens")
    assert m.feature_names == ["age"]


def test_buckley_james_categorical_formula_no_longer_crashes():
    # The centred full one-hot columns were linearly dependent, crashing
    # the WLS step with a bare LinAlgError.
    df = _df(seed=6, n=240)
    m = BuckleyJames.fit_from_df(
        df, x_col="time", formula="age + sex", c_col="cens"
    )
    assert len(m.feature_names) == 2
    assert np.all(np.isfinite(np.atleast_1d(m.params)))


def test_prediction_invariant_to_recoding():
    # The reference-coded fit spans the same model space as the old
    # one-hot fit, so predictions agree (identifiability changes the
    # parameterisation, not the fitted hazard).
    df = _df()
    new = pd.DataFrame({"age": [55.0, 45.0], "sex": ["M", "F"]})
    ref = WeibullPH.fit_from_df(
        df, x_col="time", formula="age + sex", c_col="cens"
    )
    onehot = WeibullPH.fit_from_df(
        df, x_col="time", formula="0 + age + sex", c_col="cens"
    )
    assert ref.sf([5, 10], new) == pytest.approx(
        onehot.sf([5, 10], new), rel=1e-3
    )
