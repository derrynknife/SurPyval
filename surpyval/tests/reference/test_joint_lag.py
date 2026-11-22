"""The joint model with a lag against R's ``JM`` on the Mayo PBC trial.

R 4.6.2, ``JM`` 1.5.2, ``nlme`` 3.1.171 and ``survival`` 3.8.12: the
models of ``tests/reference/test_joint_links.py`` (the log serum
bilirubin a line in years, death the event, the treatment in the hazard
only) with ``jointModel(..., lag = 1)``, printed by
``scripts/reference/reference_r_joint_lag.R``. ``JM`` evaluates the line
at ``max(t - 1, 0)``, as SurPyval does, for the value and for the area
under it (0 before the lag); its slope before the lag is the line's rate
at age 0, where SurPyval's is 0, so the slope link is not compared.

``JM`` integrates the hazard with one 15-point Gauss-Kronrod rule over
``(0, t)``, across the kink at the lag; SurPyval splits its rule there.
Every estimate agrees within 0.15 of its standard error (the largest
difference is the Weibull shape's under the cumulative link, 0.08), the
standard errors within 6% and the log-likelihoods within 0.15 (the
cumulative link's: -1999.591 here, -1999.520 in ``JM``).
"""

import warnings

import pytest

from surpyval import JointPH, Weibull
from surpyval.tests.reference.test_joint_links import _patients

# fmt: off
R = {
    "value": {
        "params": {
            "a": 0.4939272228, "b": 0.1809188048, "sd_a": 1.003071138,
            "sd_b": 0.1761373201, "corr_a_b": 0.41826033,
            "sd_noise": 0.3479324428, "alpha": 59.84512351,
            "beta": 1.038105114, "treated": 0.04336486755,
            "marker_effect": 1.286436268,
        },
        "se": {
            "a": 0.0583244, "b": 0.0130995, "treated": 0.177315,
            "marker_effect": 0.0958217,
        },
        "logLik": -1923.48967571,
    },
    "cumulative": {
        "params": {
            "a": 0.4953770169, "b": 0.1771495229, "sd_a": 0.9988330347,
            "sd_b": 0.1713458186, "corr_a_b": 0.4158243081,
            "sd_noise": 0.3489310795, "alpha": 21.69613263,
            "beta": 0.8493162615, "treated": -0.08266374848,
            "cumulative_effect": 0.1610987076,
        },
        "se": {
            "a": 0.0581054, "b": 0.0130191, "treated": 0.171889,
            "cumulative_effect": 0.0165213,
        },
        "logLik": -1999.52029981,
    },
}
# fmt: on


@pytest.mark.parametrize("name", sorted(R))
def test_a_lag_agrees_with_jm_on_the_pbc_trial(name):
    reference = R[name]
    fitter = JointPH(Weibull, link=name, lag=1.0)
    extra = {"marker_reference": 0.0} if name == "value" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fitter.fit_from_df(
            _patients(),
            x_col="years",
            c_col="censored",
            markers_col="readings",
            Z_cols=["treated"],
            hazard_covariates=["treated"],
            path_covariates=[],
            **extra,
        )
    assert model.maximum == "verified"
    assert model.parameter_names == list(reference["params"])
    se = dict(zip(model.parameter_names, model.standard_errors()))
    for parameter, estimate in zip(model.parameter_names, model.params):
        gap = (estimate - reference["params"][parameter]) / se[parameter]
        assert abs(gap) < 0.15, (parameter, gap)
    for parameter, r_se in reference["se"].items():
        assert se[parameter] == pytest.approx(r_se, rel=0.06), parameter
    assert -model.neg_ll() == pytest.approx(reference["logLik"], abs=0.15)
