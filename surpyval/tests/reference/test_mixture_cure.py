"""Mixture-cure regression (``surpyval.MixtureCure``) against stored R
``flexsurvcure`` fits (``scripts/reference/reference_r_mixture_cure.R``),
on 400 simulated units kept in the reference file.

``flexsurvcure(..., mixture = TRUE)`` maximises the same likelihood.
Its ``theta`` is the *cured* fraction and its location covariates raise
the scale, so the coefficients map as:

* logit and probit links: the incidence intercept and coefficients are
  minus ``theta``'s; the "loglog" link of ``theta`` is the complementary
  log-log of ``lfp_p`` with the same linear predictor;
* a covariate on the scale (or ``meanlog``) has minus R's coefficient
  (SurPyval's sign: positive shortens life).

flexsurv stops its BFGS search at a looser tolerance than SurPyval and
takes a numerical Hessian, so: log-likelihoods are compared to 1e-3
(absolute), estimates on the link scale to 5e-3 (absolute), standard
errors to 1% and survival probabilities to 1e-3.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import _load, values

SOURCE = "r_mixture_cure"


def _data():
    d = _load(SOURCE)["data"]
    return {k: np.asarray(v, dtype=float) for k, v in d.items()}


def _link(name, p):
    from surpyval.univariate.regression.distributional import _links

    return float(_links.link(name, p))


#: id -> (fitter, the latency parameters as flexsurv orders them with the
#: transform of each estimate, the sign of theta's linear predictor)
CASES = {
    "weibull_logit": (
        sp.MixtureCure(sp.Weibull),
        (("beta", np.log), ("alpha", np.log)),
        -1.0,
    ),
    "lnorm_probit": (
        sp.MixtureCure(sp.LogNormal, incidence="probit"),
        (("mu", float), ("sigma", np.log)),
        -1.0,
    ),
    "llogis_loglog": (
        sp.MixtureCure(sp.LogLogistic, incidence="cloglog"),
        (("beta", np.log), ("alpha", np.log)),
        1.0,
    ),
    "weibull_logit_entry": (
        sp.MixtureCure(sp.Weibull),
        (("beta", np.log), ("alpha", np.log)),
        -1.0,
    ),
}


def _fit(ref_id):
    d = _data()
    fitter = CASES[ref_id][0]
    Z = np.column_stack([d["z1"], d["z2"]])
    c = 1 - d["event"]
    if ref_id.endswith("entry"):
        keep = d["time"] > d["entry"]
        return fitter.fit(
            d["time"][keep], Z[keep], c=c[keep], tl=d["entry"][keep]
        )
    return fitter.fit(d["time"], Z, c=c)


@pytest.mark.parametrize("ref_id", sorted(CASES))
def test_fit_matches_flexsurvcure(ref_id):
    ref = values(SOURCE, ref_id)
    _, latency, sign = CASES[ref_id]
    model = _fit(ref_id)
    assert model.maximum == "verified"
    assert int(np.sum(model.data.n)) == ref["n"]
    assert_allclose(-model.neg_ll(), ref["loglik"], atol=1e-3)
    # SurPyval's maximum is at least as high
    assert -model.neg_ll() >= ref["loglik"] - 1e-6

    est = dict(zip(ref["names"], ref["est"]))
    se = dict(zip(ref["names"], ref["se"]))
    got = dict(zip(model.parameter_names, model.params))
    got_se = dict(zip(model.parameter_names, model.standard_errors()))
    link = model.incidence_link
    assert_allclose(_link(link, got["lfp_p"]), sign * est["theta"], atol=5e-3)
    (first, f_first), (second, f_second) = latency
    r_first, r_second = ref["names"][1], ref["names"][2]
    assert_allclose(f_first(got[first]), est[r_first], atol=5e-3)
    assert_allclose(f_second(got[second]), est[r_second], atol=5e-3)
    life = "mu" if "mu" in got else "alpha"
    r_life = "meanlog" if life == "mu" else "scale"
    for j, z in enumerate(("z1", "z2")):
        coef = got[f"lfp_p_coef_{j}"]
        assert_allclose(coef, sign * est[z], atol=5e-3)
        assert_allclose(got_se[f"lfp_p_coef_{j}"], se[z], rtol=1e-2)
        coef = got[f"{life}_coef_{j}"]
        assert_allclose(coef, -est[f"{r_life}({z})"], atol=5e-3)
        assert_allclose(
            got_se[f"{life}_coef_{j}"], se[f"{r_life}({z})"], rtol=1e-2
        )

    rows = np.column_stack([ref["newdata"]["z1"], ref["newdata"]["z2"]])
    for row, key in zip(rows, ("sf_row1", "sf_row2")):
        assert_allclose(model.sf(ref["times"], row), ref[key], atol=1e-3)


def test_no_covariates_on_the_cure_fraction():
    ref = values(SOURCE, "weibull_logit_scale_only")
    d = _data()
    Z = np.column_stack([d["z1"], d["z2"]])
    model = sp.WeibullMixtureCure.fit(
        d["time"], Z, c=1 - d["event"], covariates={"alpha": [0, 1]}
    )
    est = dict(zip(ref["names"], ref["est"]))
    got = dict(zip(model.parameter_names, model.params))
    assert model.parameter_names == [
        "alpha",
        "beta",
        "lfp_p",
        "alpha_coef_0",
        "alpha_coef_1",
    ]
    assert_allclose(-model.neg_ll(), ref["loglik"], atol=1e-3)
    assert_allclose(_link("logit", got["lfp_p"]), -est["theta"], atol=5e-3)
    assert_allclose(got["alpha_coef_0"], -est["scale(z1)"], atol=5e-3)
    assert_allclose(got["alpha_coef_1"], -est["scale(z2)"], atol=5e-3)
    assert_allclose(
        model.sf(ref["times"], [1, 0.5]), ref["sf_row2"], atol=1e-3
    )
