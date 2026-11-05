"""Aalen's additive hazards model with sampling weights, checked against R.

The reference is R 4.6.2, ``timereg`` 2.0.7:
``aalen(Surv(x, d) ~ z1 + z2, data = df, weights = df$w, robust = 1)``,
and the same with delayed entry, ``Surv(tl, x, d)``, on the forty
weighted reference rows (``_helpers.weighted_reference_data``), printed
by ``scripts/reference/reference_r_weighted_aalen.R``: the cumulative
regression functions ``cum`` and their robust variances ``robvar.cum`` at
five times. ``timereg`` breaks tied times at random, so the script and
this test first break them the same way, adding ``0.001 * (rank within
the tie - 1)`` after sorting by time, then events first, then row.

Tolerance: ``1e-12`` relative (with ``1e-15`` absolute) on both, since
both sides are the same closed form (weighted least squares at each
event time, and the sum of the units' squared influences); they agree to
about ``1e-15``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data

TIMES = [0.5, 1.5, 3.0, 5.0, 7.0]

# fmt: off
R = {
    "weighted": (
        [[0.000220939890328192, 0.046446786837849002, -0.003966128557790630],
         [-0.001694588478566873, 0.233481142302845279, 0.030419838393887857],
         [0.340394761890076669, 0.389373729823305592, 0.381260398550503588],
         [0.654447019617653947, 0.663095895889474929, 0.615658705434066422],
         [1.032575435232150785, 0.903921279735037997, 0.886610328527973612]],
        [[8.71048908234483e-07, 2.09966710741274e-03, 5.26383406428492e-05],
         [4.49755648942830e-05, 1.26552083390833e-02, 1.07640362258538e-03],
         [2.16188586107775e-02, 6.80203375640801e-02, 1.49835228110104e-02],
         [4.04129743155997e-02, 2.35869937565879e-01, 4.03962118223664e-02],
         [7.20891859786492e-02, 2.78479361351239e-01, 3.73918372623159e-02]],
    ),
    "weighted_tl": (
        [[1.22709386712485e-03, 8.43211602373062e-02, -1.20150244942174e-02],
         [-9.32499322374141e-05, 2.76332183063297e-01, 2.31402330369189e-02],
         [3.41996100436406e-01, 4.32224770583757e-01, 3.73980793193534e-01],
         [6.56048358163983e-01, 7.05946936649927e-01, 6.08379100077097e-01],
         [1.03417677377848e+00, 9.46772320495490e-01, 8.79330723171003e-01]],
        [[6.84877919050142e-06, 6.65569727114764e-03, 2.66503565990034e-04],
         [4.34958980857499e-05, 1.71180756344890e-02, 1.37201076265054e-03],
         [2.16237947719520e-02, 7.50254099430334e-02, 1.55803657570778e-02],
         [4.04213319038936e-02, 2.48585007236011e-01, 4.22999850826207e-02],
         [7.21389036450207e-02, 2.92718375038130e-01, 3.94483890811873e-02]],
    ),
}
# fmt: on


def _untied():
    """The reference rows sorted by time, then events first, then row,
    with each tie broken by 0.001 * (rank within the tie - 1)."""
    data = weighted_reference_data()
    k = data["x"].size
    order = np.lexsort((np.arange(k), -data["d"], data["x"]))
    x = data["x"][order]
    _, first = np.unique(x, return_index=True)
    rank = np.arange(k) - np.repeat(first, np.diff(np.r_[first, k]))
    out = {key: value[order] for key, value in data.items()}
    out["x"] = x + 0.001 * rank
    return out


@pytest.mark.parametrize("case", sorted(R))
def test_weighted_aalen_matches_timereg(case):
    data = _untied()
    model = sp.AalenAdditive.fit(
        data["x"],
        data["Z"],
        c=data["c"],
        w=data["w"],
        tl=data["tl"] if case == "weighted_tl" else None,
    )
    cum, robvar = R[case]
    assert model.weighted
    assert_allclose(
        model.cumulative_coefficients(TIMES).values,
        cum,
        rtol=1e-12,
        atol=1e-15,
    )
    t = np.asarray(TIMES)
    assert_allclose(model._B_var_at(t), robvar, rtol=1e-12, atol=1e-15)
    # The bounds and the summary take the robust variance
    se = np.sqrt(np.asarray(robvar)[-1])
    assert_allclose(model.summary(at=7.0)["se(coef)"], se, rtol=1e-12)
