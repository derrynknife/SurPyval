"""The joint model's links against R's ``JM`` on the Mayo PBC trial.

R 4.6.2, ``JM`` 1.5.2, ``nlme`` 3.1.171 and ``survival`` 3.8.12:
``jointModel(lme(log(serBilir) ~ year, random = ~ year | id),
coxph(Surv(years, status2) ~ treated), timeVar = "year", method =
"weibull-PH-aGH")``, with ``parameterization`` and ``derivForm`` for the
slope, the value and slope together, and the area under the line (the
cumulative parameterization), printed by
``scripts/reference/reference_r_joint_links.R``.

The data are ``load_pbc2``: the marker is the log of the serum bilirubin,
a line in years with a level and a rate of each patient's own; the event
is death; the treatment enters the hazard only. ``JM``'s Weibull is
``sigma.t t^(sigma.t - 1) exp(g0 + ...)``: the script prints it as
SurPyval's ``alpha`` (at a marker of 0) and ``beta``.

The two programs maximise the same likelihood by different quadratures:
``JM`` places its Gauss-Hermite nodes once, from the linear mixed model's
fit of the readings alone (pseudo-adaptive), and SurPyval places them at
the mode of each patient's whole integrand, again at every round. So the
estimates agree to a fraction of their standard errors, not to rounding:
every one within 0.15 standard errors here (the largest differences are
the Weibull shape's, 0.11 of its standard error, for the slope and for
the cumulative link), the standard errors within 6% of each other, and
the log-likelihoods within 0.15.

The largest gap in the log-likelihood is the cumulative link's: ``JM``
reports -1985.683 at its estimates and SurPyval -1985.805 at its own. An
independent integral (a grid of 101 points a dimension over each
patient's level and rate, nine standard deviations each way about their
values given the readings, and Simpson's rule with 401 points in age)
gives -1985.8053 at SurPyval's estimates and -1985.8119 at ``JM``'s: the
likelihood SurPyval reports is right to 4e-5, and its point is the
higher of the two.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from surpyval import JointPH, Weibull
from surpyval.datasets import load_pbc2

# fmt: off
R = {
    "value": {
        "link": "value",
        "params": {
            "a": 0.4927791355, "b": 0.1850057222, "sd_a": 1.002452357,
            "sd_b": 0.180766219, "corr_a_b": 0.4252965087,
            "sd_noise": 0.3471270622, "alpha": 75.77330815,
            "beta": 1.018957464, "treated": 0.04397101175,
            "marker_effect": 1.240054997,
        },
        "se": {
            "a": 0.0582897, "b": 0.0133274, "treated": 0.179148,
            "marker_effect": 0.0931885,
        },
        "logLik": -1919.22101624,
    },
    "slope": {
        "link": "slope",
        "params": {
            "a": 0.483322542, "b": 0.2197284205, "sd_a": 0.9682163831,
            "sd_b": 0.2122601976, "corr_a_b": 0.6848956976,
            "sd_noise": 0.3543675047, "alpha": 33.28675913,
            "beta": 2.137459518, "treated": 0.04249358222,
            "slope_effect": 10.77569256,
        },
        "se": {
            "a": 0.0563481, "b": 0.0146209, "treated": 0.21742,
            "slope_effect": 1.20502,
        },
        "logLik": -1940.33882208,
    },
    "value+slope": {
        "link": ["value", "slope"],
        "params": {
            "a": 0.4894671189, "b": 0.1959833549, "sd_a": 0.9986702379,
            "sd_b": 0.1893867095, "corr_a_b": 0.4866900878,
            "sd_noise": 0.3470023717, "alpha": 78.60145629,
            "beta": 1.174664715, "treated": 0.03645750719,
            "marker_effect": 1.036371945, "slope_effect": 2.894945077,
        },
        "se": {
            "a": 0.0580799, "b": 0.0141123, "treated": 0.187227,
            "marker_effect": 0.122215, "slope_effect": 0.994678,
        },
        "logLik": -1914.52485619,
    },
    "cumulative": {
        "link": "cumulative",
        "params": {
            "a": 0.4945570565, "b": 0.1779626562, "sd_a": 0.9993626123,
            "sd_b": 0.1726138087, "corr_a_b": 0.4165108395,
            "sd_noise": 0.3487046648, "alpha": 27.85724664,
            "beta": 0.7921695833, "treated": -0.0980668379,
            "cumulative_effect": 0.1575167404,
        },
        "se": {
            "a": 0.0581327, "b": 0.0130584, "treated": 0.172947,
            "cumulative_effect": 0.0144275,
        },
        "logLik": -1985.68286974,
    },
}
# fmt: on
# The standard errors compared are those both programs give on the same
# scale: the line's means, the treatment's and the links' effects.


def _patients():
    pbc = load_pbc2()
    pbc["y"] = np.log(pbc["serBilir"])
    rows = [
        {
            "years": visits["years"].iloc[0],
            "censored": 1 - visits["status2"].iloc[0],
            "treated": float(visits["drug"].iloc[0] == "D-penicil"),
            "readings": dict(zip(visits["year"], visits["y"])),
        }
        for _, visits in pbc.groupby("id")
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name", sorted(R))
def test_the_links_agree_with_jm_on_the_pbc_trial(name):
    reference = R[name]
    fitter = JointPH(Weibull, link=reference["link"])
    extra = {"marker_reference": 0.0} if "value" in fitter.link else {}
    # the value link's fit of these data stops with its gradient at 3e-4
    # standard errors, above the 1e-4 of a verified maximum, and says so
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
    if name != "value":
        assert model.maximum == "verified"
    assert model.parameter_names == list(reference["params"])
    se = dict(zip(model.parameter_names, model.standard_errors()))
    for parameter, estimate in zip(model.parameter_names, model.params):
        gap = (estimate - reference["params"][parameter]) / se[parameter]
        assert abs(gap) < 0.15, (parameter, gap)
    for parameter, r_se in reference["se"].items():
        assert se[parameter] == pytest.approx(r_se, rel=0.06), parameter
    assert -model.neg_ll() == pytest.approx(reference["logLik"], abs=0.15)
