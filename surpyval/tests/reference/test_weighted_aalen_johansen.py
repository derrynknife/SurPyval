"""The Aalen-Johansen cumulative incidence of data with sampling weights
(``CompetingRisks.fit(..., w=)``) against R ``survival``.

R 4.6.2 and ``survival`` 3.8.12:
``summary(survfit(Surv(x, factor(e, 0:2)) ~ 1, weights = w), times =
TIMES, extend = TRUE)``, its ``pstate`` and ``std.err`` (the
infinitesimal-jackknife, robust, standard error; the same with ``id =`` and
``robust = TRUE``), printed by
``scripts/reference/reference_r_weighted_aalen_johansen.R``. The data are
``_helpers.weighted_reference_data()`` (forty rows, ties among the failures
and between failures and censorings), each failure from cause 1 where
``z1 = 1`` and cause 2 otherwise; "unit" is the same with every weight 1
(the robust standard error of the unweighted estimate, which is not
Aalen's); "ends" adds a unit failing from cause 1 at 15 (weight 1.1), after
every other time, so that the last step leaves no one at risk.

Tolerance: closed forms, equal to rounding (``rtol=1e-10``).
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data

TIMES = np.array([1, 2.2, 3.2, 4.3, 6.1, 8.3, 9.6, 15])

# fmt: off
R = {
    "weighted": {
        "cif1": [0.0768388106416276, 0.137871674491393, 0.209127102655812, 0.291176323163342, 0.299541968758639, 0.451356723099547, 0.451356723099547, 0.451356723099547],  # noqa: E501
        "se1": [0.03925988299114, 0.0583067688552122, 0.0713446733348122, 0.0821123411051695, 0.0823656451773554, 0.103506713266834, 0.103506713266834, 0.103506713266834],  # noqa: E501
        "cif2": [0, 0.0856025039123631, 0.197552698828461, 0.258473088956523, 0.258473088956523, 0.281499368087736, 0.281499368087736, 0.281499368087736],  # noqa: E501
        "se2": [0, 0.0482450479083277, 0.0731643391951063, 0.0799800715331592, 0.0799800715331592, 0.0820150338733892, 0.0820150338733892, 0.0820150338733892],  # noqa: E501
    },
    "unit": {
        "cif1": [0.1, 0.15, 0.201666666666667, 0.290351751207729, 0.325919384057971, 0.46253142841458, 0.46253142841458, 0.46253142841458],  # noqa: E501
        "se1": [0.0474341649025257, 0.0564579489531811, 0.0637582601765508, 0.0744534805478091, 0.078658829765951, 0.0928736844084879, 0.0928736844084879, 0.0928736844084879],  # noqa: E501
        "cif2": [0, 0.075, 0.1525, 0.211701388888889, 0.211701388888889, 0.253735864075538, 0.253735864075538, 0.253735864075538],  # noqa: E501
        "se2": [0, 0.0416458281223942, 0.0573517109596567, 0.0669048039595479, 0.0669048039595479, 0.0751836733379038, 0.0751836733379038, 0.0751836733379038],  # noqa: E501
    },
    "ends": {
        "cif1": [0.0755384615384615, 0.135538461538462, 0.205570065355466, 0.285722327994729, 0.293818262968652, 0.437684936933383, 0.437684936933383, 0.724351270667781],  # noqa: E501
        "se1": [0.0385947989761385, 0.0573584595340213, 0.0702261637300617, 0.0806980205678396, 0.0809276626152828, 0.100812952481512, 0.100812952481512, 0.0804901796989341],  # noqa: E501
        "cif2": [0, 0.0841538461538462, 0.194181277039674, 0.253708926822071, 0.253708926822071, 0.275648729332218, 0.275648729332218, 0.275648729332218],  # noqa: E501
        "se2": [0, 0.047449958292031, 0.072032678650487, 0.0786601304300033, 0.0786601304300033, 0.0804901796989341, 0.0804901796989341, 0.0804901796989341],  # noqa: E501
    },
}
# fmt: on


def _data(case):
    data = weighted_reference_data()
    x, c, w = data["x"], data["c"], data["w"]
    e = np.where(c == 0, np.where(data["Z"][:, 0] == 1, 1, 2), None)
    if case == "unit":
        w = np.ones_like(w)
    if case == "ends":
        x, c, w = np.append(x, 15.0), np.append(c, 0), np.append(w, 1.1)
        e = np.append(e, 1)
    return x, c, e, w


def _se(model, cause, times):
    idx = np.searchsorted(model.x, times, side="right") - 1
    var = model.cif_var[model.event_idx_map[cause]]
    return np.where(idx >= 0, np.sqrt(var[np.maximum(idx, 0)]), 0.0)


@pytest.mark.parametrize("case", sorted(R))
def test_cumulative_incidence_and_robust_se_match_survfit(case):
    x, c, e, w = _data(case)
    model = sp.CompetingRisks.fit(x, e, c, w=w)
    # R extends the curve past the last time; SurPyval's cb does not.
    times = TIMES[TIMES <= model.x[-1]]
    k = times.size
    for cause in (1, 2):
        assert_allclose(
            model.cif(times, cause), R[case][f"cif{cause}"][:k], rtol=1e-10
        )
        assert_allclose(
            _se(model, cause, times),
            R[case][f"se{cause}"][:k],
            rtol=1e-10,
            atol=1e-14,
        )
        # The normal bounds are the estimate plus or minus z times it
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cb = model.cb(times, cause, bound_type="normal")
        half = 1.959963984540054 * np.asarray(R[case][f"se{cause}"][:k])
        assert_allclose(cb[:, 1] - cb[:, 0], 2 * half, rtol=1e-9, atol=1e-14)
