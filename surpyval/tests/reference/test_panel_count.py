"""The panel-count models against R's ``spef`` on simulated panel counts
(the shared fixture ``panel_counts``: 85 units seen at their own visits,
a treatment indicator and two baseline covariates;
``scripts/reference/reference_r_spef.R`` wrote the values with spef
1.0.9).

The maximum pseudo-likelihood fits (with and without covariates) are the
same estimator as spef's ``"MPL"``, run with ``absTol = relTol = 1e-12``
(at the default tolerances the two optimisers stop at slightly different
points, so they are compared at the tighter setting), and agree to the
precision of the searches. spef has no step-function maximum likelihood fit
with covariates: its ``"MLs"`` maximises the same likelihood over a monotone
spline baseline (Lu, Zhang and Huang 2007), a smaller family than the step
functions, so it is compared with a tolerance and by the likelihood
(SurPyval's maximum is at least as high as the likelihood at spef's
coefficients).
"""

import numpy as np
import pandas as pd

from surpyval.recurrent import PanelCountMCF, PanelCountRegression
from surpyval.recurrent import panel_count as pc

from ._data import fixture, values

Z_COLS = ["treatment", "size", "num"]


def _df():
    return pd.DataFrame(fixture("panel_counts"))


def _fit(how):
    return PanelCountRegression.fit_from_df(
        _df(), "time", Z_COLS, "count", i_col="id", how=how
    )


def test_pseudo_likelihood_regression_matches_spef_mpl():
    ref = values("r_spef", "mpl")
    model = _fit("pseudo")
    np.testing.assert_allclose(model.params, ref["coef"], rtol=1e-6)
    np.testing.assert_allclose(
        model.baseline_mcf(ref["times"]), ref["baseline"], rtol=1e-6
    )


def test_pseudo_likelihood_mcf_matches_spef_isotonic_regression():
    ref = values("r_spef", "mpl_nonparametric")
    model = PanelCountMCF.fit_from_df(
        _df(), "time", "count", i_col="id", how="pseudo"
    )
    np.testing.assert_allclose(model.mcf(ref["times"]), ref["mcf"], rtol=1e-12)


def test_likelihood_regression_is_near_spef_sieve_mle_and_higher():
    ref = values("r_spef", "mls")
    model = _fit("MLE")
    # The sieve's spline baseline is a smaller family than the step
    # functions: close coefficients, not equal ones.
    np.testing.assert_allclose(model.params, ref["coef"], atol=0.01)

    df = _df()
    panel = pc._panel_data(df["time"], df["id"], df["count"], df[Z_COLS])
    parts = pc._likelihood_parts(panel)

    def profile(beta):
        eta = panel.Z @ beta
        delta, ok = pc._full_mean(panel, np.exp(eta))
        assert ok
        return pc._full_loglik(panel, delta, np.exp(eta), parts) + float(
            (panel.Z.T @ panel.total) @ beta
        )

    assert profile(model.params) >= profile(ref["coef"])
