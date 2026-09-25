"""
Tests for the nonparametric competing-risks (Aalen-Johansen) estimator.

Covers the #253 fixes: the incidence increment must weight the
cause-specific hazard by the survival just *before* each event time
(S(t-)), queries before the first observed time must return the
zero/one boundary values instead of wrapping to the last step, and
``fit_from_df`` must not shadow the ``df`` (density) method.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval.univariate.competing_risks.nonparametric.competing_risks import (  # noqa: E501
    CompetingRisks,
)


def test_single_cause_km_cif_reaches_one():
    # With one cause and no censoring the KM-weighted Aalen-Johansen CIF
    # is 1 - KM, which reaches exactly 1 at the last event time.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = np.array(["a"] * 5)
    cr = CompetingRisks.fit(x=x, e=e, method="Kaplan-Meier")
    assert cr.cif(np.array([5.0]), "a")[0] == pytest.approx(1.0, abs=1e-12)


def test_two_cause_cifs_sum_to_one_minus_km():
    rng = np.random.default_rng(1)
    n = 400
    t1 = rng.weibull(2, n) * 10
    t2 = rng.weibull(1.5, n) * 12
    t = np.minimum(t1, t2)
    ev = np.where(t1 < t2, "a", "b")
    cens = t > 15
    tt = np.minimum(t, 15.0)
    c = cens.astype(int)
    e = np.array(
        [ev[i] if not cens[i] else None for i in range(n)], dtype=object
    )
    cr = CompetingRisks.fit(x=tt, e=e, c=c, method="Kaplan-Meier")

    q = np.array([2.0, 5.0, 10.0, 14.0])
    total = cr.cif(q, "a") + cr.cif(q, "b")
    idx = np.searchsorted(cr.x, q, side="right") - 1
    assert np.allclose(total, 1.0 - cr.S[idx], atol=1e-12)


def test_queries_before_first_event_are_boundary_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = np.array(["a"] * 5)
    cr = CompetingRisks.fit(x=x, e=e)
    assert cr.sf(np.array([0.1]))[0] == 1.0
    assert cr.Hf(np.array([0.1]))[0] == 0.0
    assert cr.hf(np.array([0.1]))[0] == 0.0
    assert cr.cif(np.array([0.1]), "a")[0] == 0.0
    assert cr.iif(np.array([0.1]), "a")[0] == 0.0


def test_fit_from_df_does_not_shadow_density_method():
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": ["a", "b", "a", "b"],
        }
    )
    model = CompetingRisks.fit_from_df(frame, x_col="time", e_col="event")
    assert model.source_df is frame
    # ``df`` must still be the density method.
    vals = model.df(np.array([2.5]))
    assert np.isfinite(vals).all()


def test_the_requested_survival_estimator_is_reported():
    # method only changed the stored S array; sf/ff/Hf always used exp(-H)
    import json

    from surpyval import KaplanMeier, NelsonAalen
    from surpyval.univariate.competing_risks import CompetingRisks

    rng = np.random.default_rng(0)
    t1 = rng.exponential(2.0, 60)
    t2 = rng.exponential(3.0, 60)
    x = np.minimum(t1, t2).round(1)
    e = np.where(t1 < t2, 1, 2)
    q = np.array([0.5, 1.0, 2.0])

    km = CompetingRisks.fit(x, e, method="Kaplan-Meier")
    na = CompetingRisks.fit(x, e)
    assert np.allclose(km.sf(q), KaplanMeier.fit(x).sf(q))
    assert np.allclose(na.sf(q), NelsonAalen.fit(x).sf(q))
    assert np.allclose(km.ff(q), 1 - km.sf(q))
    assert np.allclose(km.sf(q), np.exp(-km.Hf(q)))
    # one cause's (net) survival: product limit of its own increments
    d1 = np.array([(x[e == 1] == t).sum() for t in km.x])
    net = np.cumprod(1 - d1 / km.r)
    assert np.allclose(km.sf(km.x, 1), net)

    restored = CompetingRisks.from_dict(json.loads(json.dumps(km.to_dict())))
    assert restored.method == "Kaplan-Meier"
    assert np.allclose(restored.sf(q, 2), km.sf(q, 2))
